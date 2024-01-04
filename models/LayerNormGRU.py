import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNormGRUCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LayerNormGRUCell, self).__init__()

        self.ln_i2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_h2h = torch.nn.LayerNorm(2*hidden_size, elementwise_affine=False)
        self.ln_cell_1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln_cell_2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False)

        self.i2h = torch.nn.Linear(input_size, 2 * hidden_size, bias=bias)
        self.h2h = torch.nn.Linear(hidden_size, 2 * hidden_size, bias=bias)
        self.h_hat_W = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.h_hat_U = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, h):

        h = h.view(h.size(0), -1)
        x = x.view(x.size(0), -1)

        # Linear mappings
        i2h = self.i2h(x)
        h2h = self.h2h(h)

        # Layer norm
        i2h = self.ln_i2h(i2h)
        h2h = self.ln_h2h(h2h)

        preact = i2h + h2h

        # activations
        gates = preact[:, :].sigmoid()
        z_t = gates[:, :self.hidden_size]
        r_t = gates[:, -self.hidden_size:]

        # h_hat
        h_hat_first_half = self.h_hat_W(x)
        h_hat_last_half = self.h_hat_U(h)

        # layer norm
        h_hat_first_half = self.ln_cell_1( h_hat_first_half )
        h_hat_last_half = self.ln_cell_2( h_hat_last_half )

        h_hat = torch.tanh(  h_hat_first_half + torch.mul(r_t,   h_hat_last_half ) )

        h_t = torch.mul( 1-z_t , h ) + torch.mul( z_t, h_hat)

        # Reshape for compatibility

        h_t = h_t.view( h_t.size(0), -1)
        return h_t


class LayerNormGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers = 2, bias=True):
        super(LayerNormGRU, self).__init__()

        self.input_dim = input_dim
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        self.hidden0 = nn.ModuleList([
            LayerNormGRUCell(input_size=(input_dim if layer == 0 else hidden_dim), hidden_size=hidden_dim, bias=bias)
            for layer in range(num_layers)
        ])


    def forward(self, input: torch.Tensor, seq_lens=None):
        seq_len, batch_size, _ = input.size()
        hx = input.new_zeros(self.num_layers, batch_size, self.hidden_dim, requires_grad=False)

        ht = []
        for i in range(seq_len):
            ht.append([None] * (self.num_layers))

        seq_len_mask = input.new_ones(batch_size, seq_len, self.hidden_dim, requires_grad=False)
        if seq_lens != None:
            for i, l in enumerate(seq_lens):
                seq_len_mask[i, l:, :] = 0
        seq_len_mask = seq_len_mask.transpose(0, 1)

        indices = (torch.cuda.LongTensor(seq_lens) - 1).unsqueeze(1).unsqueeze(0).unsqueeze(0).repeat(
            [1, self.num_layers, 1, self.hidden_dim])
        h = hx

        for t, x in enumerate(input):
            for l, layer in enumerate(self.hidden0):
                ht_= layer(x, h[l])
                ht[t][l] = ht_ * seq_len_mask[t]
                x = ht[t][l]
            ht[t] = torch.stack(ht[t])
            h = ht[t]
        y = torch.stack([h[-1] for h in ht])
        hy = torch.stack(list(torch.stack(ht).gather(dim=0, index=indices).squeeze(0)))

        return y, hy
        # seq_len, batch_size, _ = input.size()
        # # hidden
        # h0 = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
        #
        # outs = []
        #
        # hn_1 = h0[0, :, :]
        # hn_2 = h0[1, :, :]
        #
        # for seq in range(seq_len):
        #     hn_1 = self.gru_cell_1(input[seq, :, :], hn_1)
        #     hn_2 = self.gru_cell_2(hn_1, hn_2)
        #     outs.append(hn_2)
        #
        # out = outs[-1].squeeze()
        # return out
        # # Initialize hidden state with zeros
        # #######################
        # #  USE GPU FOR MODEL  #
        # #######################
        # # print(x.shape,"x.shape")100, 28, 28
        # if torch.cuda.is_available():
        #     h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        # else:
        #     h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        #
        # outs = []
        #
        # hn = h0[0, :, :]
        #
        # for seq in range(x.size(1)):
        #     hn = self.grus(x[:, seq, :], hn)
        #     outs.append(hn)
        #
        # out = outs[-1].squeeze()
        #
        # return out

'''
test the module
'''
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
def is_equal(a, b, epsilon=1e-5):
    return torch.all(torch.lt(torch.abs(torch.add(a, -b)), epsilon)).item() == 1

def test_layernorm_LSTMCell():
    batch_size = 4
    hidden_size = 2
    num_input_features = 3
    # create two objects
    rnn = LayerNormGRUCell(num_input_features, hidden_size, bias=True)
    rnn_old = torch.nn.GRUCell(num_input_features, hidden_size, bias=True)
    # initialize two objects with same weights & biases
    for param in rnn_old.named_parameters():
        rnn.register_parameter(param[0], param[1])
    # initialize the hidden state
    states = torch.tensor(torch.zeros(batch_size, hidden_size,1))
    # create the input data
    input_tensor = torch.FloatTensor(np.random.rand(batch_size, num_input_features))

    # normal operation for use LSTM to decode the data
    rnn_old_h, rnn_old_c = rnn_old(input_tensor, states)
    # use the new LSTM to decode the data
    rnn_h, rnn_c = rnn(input_tensor, states)

    # check whether the two objects' outputs are the same
    print("whether the two objects' h_1 are the same: ", is_equal(rnn_old_h, rnn_h))
    print("whether the two objects' c_1 are the same: ", is_equal(rnn_old_c, rnn_c))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size)
    f = torch.matmul(rnn_h, x)
    f.backward(torch.ones(batch_size))
    print("the backward operation can be run normally")

def test_layernorm_LSTM(use_biLSTM=True):
    batch_size = 4
    max_length = 3
    hidden_size = 2
    n_layer = 5
    num_input_features = 3
    n_direction = 2 if use_biLSTM else 1
    # create two objects
    rnn = LayerNormLSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM, use_layer_norm=False)
    rnn_old = torch.nn.LSTM(num_input_features, hidden_size, n_layer, bias=True, bidirectional=use_biLSTM)
    # initialize two objects with same weights
    rnn.copy_parameters(rnn_old)
    # initialize the hidden state
    states = (torch.zeros(n_layer*n_direction, batch_size, hidden_size), torch.zeros(n_layer*n_direction, batch_size, hidden_size))
    # create the sequence data with padding
    input_tensor = torch.zeros(batch_size, max_length, num_input_features)
    input_tensor[0] = torch.FloatTensor([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    input_tensor[1] = torch.FloatTensor([[4, 5, 6], [5, 7, 8], [0, 0, 0]])
    input_tensor[2] = torch.FloatTensor([[6, 4, 3], [8, 1, 9], [0, 0, 0]])
    input_tensor[3] = torch.FloatTensor([[7, 3, 5], [0, 0, 0], [0, 0, 0]])
    seq_lengths = [3, 2, 2, 1]
    # transform the sequence data into new shape [max_length, batch_size, num_input_features]
    batch_in = Variable(input_tensor)
    batch_in = batch_in.permute(1, 0, 2)
    # normal operation for use LSTM to decode the sequence
    pack = torch.nn.utils.rnn.pack_padded_sequence(batch_in, seq_lengths)
    rnn_old_out, rnn_old_states = rnn_old(pack, states)
    rnn_old_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_old_out)
    # use the new LSTM to decode the sequence
    rnn_out, rnn_states = rnn(batch_in, states, seq_lengths)

    # check whether the two objects' outputs are the same
    print("whether the two objects' outputs are the same: ", is_equal(rnn_old_out, rnn_out))
    print("whether the two objects' h_n are the same: ", is_equal(rnn_old_states[0], rnn_states[0]))
    print("whether the two objects' c_n are the same: ", is_equal(rnn_old_states[1], rnn_states[1]))

    # check whether the gradient backward can be done
    x = torch.ones(hidden_size * n_direction)
    f = torch.matmul(rnn_out, x)
    f.backward(torch.ones(max_length, batch_size))
    print("the backward operation can be run normally")


if __name__ == "__main__":
    print("start checking the layernorm-LSTMCell......")
    test_layernorm_LSTMCell()
    print()
    print("start checking the layernorm-LSTM......")
    test_layernorm_LSTM(use_biLSTM=False)
    print()
    print("start checking the bi-layernorm-LSTM......")
    test_layernorm_LSTM(use_biLSTM=True)
