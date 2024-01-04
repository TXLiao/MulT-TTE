import copy
import time
from typing import Dict
import gc

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, to_var


def train_model(model: nn.Module, data_loaders: Dict[str, DataLoader],
                loss_func: callable, optimizer: optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    num_epochs = args.epochs
    beta = args.beta
    phases = ['train', 'val', 'test']
    since = time.clock()

    with open(model_folder + "/output.txt", "a") as f:
        f.write(str(model))
        f.write("\n\n")

    save_dict, best_mae = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 10000

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=2,
                                                     threshold=1e-2, threshold_mode='rel', min_lr=1e-7, verbose=True)

    try:
        patiance = 0
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss = {phase: 0.0 for phase in phases}
            msg = []
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                steps, predictions, targets = 0, list(), list()
                tqdm_loader = tqdm(data_loaders[phase],mininterval=3)
                for step, (features, truth_data) in enumerate(tqdm_loader):
                    steps += truth_data.size(0)
                    features = to_var(features, args.device)
                    targets.append(truth_data.numpy())
                    truth_data = to_var(truth_data, args.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):

                        outputs, loss_1 = model(features, args)
                        loss_2 = loss_func(truth=truth_data, predict=outputs)
                        loss = (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() + beta * loss_2

                        tqdm_loader.set_description(
                            f'{phase} epoch: {epoch}, {phase} loss: {(running_loss[phase] / steps) :.8f}, '
                            f'loss1: {loss_1.item()}, loss2: {loss_2.item()}')

                        if phase == 'train':
                            loss.backward()
                            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 50)  # after 50  # 20效果不佳，无法达到最优
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().detach().numpy())

                    running_loss[phase] += loss.item() * truth_data.size(0)
                    if step % 1000 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                torch.cuda.empty_cache()

                predictions = np.concatenate(predictions).copy()
                targets = np.concatenate(targets).copy()
                scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                           targets.reshape(targets.shape[0], -1), args, plot=epoch % 5 == 0, **kwargs)
                with open(model_folder+"/output.txt", "a") as f:
                    f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n')
                    f.write(str(scores))
                    f.write('\n')
                    f.write(str(time.time()))
                    f.write("\n\n")
                print(scores)
                msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}\n {scores}\n")
                if phase == 'val':
                    if scores['MAE'] < best_mae:
                        best_mae = scores['MAE']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(f"{model_folder}/best_model.pkl", **save_dict)
                        patiance = 0
                    else:
                        patiance += 1
                        print(f"Current MAE {scores['MAE']} more than best MAE {best_mae}, patience: {patiance}")

            if patiance >= args.patience:
                print(f"Early stop! best MAE: {best_mae}")
                break
            scheduler.step(running_loss['val'])

    finally:
        time_elapsed = time.clock() - since
        print(f"cost {time_elapsed} seconds")

        save_model(f"{model_folder}/best_model.pkl", **save_dict)
        save_model(f"{model_folder}/final_model.pkl",
                   **{'model_state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': epoch,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})
