import sys
import os
import argparse
from train.training_main import train_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='input the model name', default='MulT_TTE', choices=['MulT_TTE'])
    parser.add_argument('-M', '--mode', type=str, default='train', help='input the process mode', choices=['train', 'resume', 'test'])
    parser.add_argument('-d', '--dataset', type=str, default='porto_MulT_TTE', help='input the dataset name', choices=['chengdu_MulT_TTE','porto_MulT_TTE'])
    parser.add_argument('-i', '--identify', type=str, help='input the specific identification information', default='')

    parser.add_argument('-D', '--device', type=str, help='input the chosen device', default="cuda:0")

    parser.add_argument('-o', '--optim', type=str, help='input the chosen optimization function', default="Adam", choices=['Adam'])

    parser.add_argument('-c', '--loss', type=str, help='input the chosen loss function', default="smoothL1", choices=['rmse','mse', 'mape', 'mae', 'smoothL1'])
    parser.add_argument('-cl', '--loss_val', type=float, help='intput the specific parameter for smoothL1',  default=300.)
    parser.add_argument('-e', '--epochs', type=int, help='input the max epochs',default=50)
    parser.add_argument('-b', '--beta', type=float, help='intput the learning preference between MSG and TTE (the bigger the value, the more preference for TTE.)',default=0.7)
    parser.add_argument('-l', '--lr', type=float, help='intput the initial learning rate',default=0.001)
    parser.add_argument('-w', '--weight_decay', type=float, help='intput the weight decay of optimization',default=0.00001)
    parser.add_argument('-p', '--patience', type=int, help='intput the max iteration times of early stop',default=10)
    parser.add_argument('-r', '--mask_rate', type=float, help='intput the mask rate of segments in a trajectory', default=0.4)

    args = parser.parse_args()
    args.absPath = os.path.dirname(os.path.abspath(__file__))
    print(args.model)
    print(args.dataset)
    train_main(args)
    sys.exit(0)