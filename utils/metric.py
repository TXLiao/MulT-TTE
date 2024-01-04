import os.path

import numpy as np
from scipy.stats import pearsonr


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        mse = np.nan_to_num(mse)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        mae = np.nan_to_num(mae)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
        mape = np.nan_to_num(mask * mape)
        mape = np.nan_to_num(mape)
        return np.mean(mape)


def calculate_metrics(preds, labels, args = None, null_val=0.0, plot=False, inds=None):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    try:

        preds = preds.reshape([-1,1]).squeeze()
        labels = labels.reshape([-1,1]).squeeze()
        print(f"preds: {preds[:40000:1905]}")
        print(f"label: {labels[:40000:1905]}")
        mape = np.mean(np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5)))
        mse = np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.subtract(preds, labels)).astype('float32'))

    except Exception as e:
        print(e)
        mae = 0
        mape = 0
        rmse = 0
    try:
        pearsonrs = pearsonr(preds, labels)
    except Exception as e:
        print(e)
        pearsonrs = (None,None)
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'PEARR':pearsonrs[0], 'PEARP': pearsonrs[1]}


def cal_acc(input_score, target): #np.argmax(output[1][0][7].cpu().detach().numpy()),labels[0][7]
    sizes = input_score.shape
    count, pred_true = 0, 0
    for batch in range(sizes[0]):
        for idx in range(sizes[1]):
            if target[batch][idx] == -100: continue
            count += 1
            if np.argmax(input_score[batch][idx].cpu().detach().numpy()) == target[batch][idx]:
                pred_true += 1
                # if pred_true % 5000 == 0:
                #     print(f"pred: {np.argmax(input_score[batch][idx].cpu().detach().numpy())}, true: {target[batch][idx]}")

    return pred_true / count, pred_true, count

