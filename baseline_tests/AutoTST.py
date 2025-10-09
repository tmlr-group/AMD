import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader import load_data
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
import torch

def wildbootstrapWD2(test_stat, hh1, hh0, n_res, device, dtype):
    N1 = len(hh1)
    stats = []
    for k in range(n_res):
        weights = np.random.exponential(scale=1, size = N1)
        weights = torch.tensor(weights/np.mean(weights), device=device, dtype=dtype).reshape(1,-1)
        hh1_ = weights.t() @ weights * hh1
        stat1 = (torch.sum(hh1_)- hh1_.trace())/ (N1*(N1-1))
        stat = stat1.item()
        
        if hh0 is not None:
            weights = np.random.exponential(scale=1, size = N1)
            weights = torch.tensor(weights/np.mean(weights), device=device, dtype=dtype).reshape(1,-1)
            hh0_ = weights.t() @ weights * hh0
            stat0 = (torch.sum(hh0_)- hh0_.trace())/ (N1*(N1-1))
            stat -= stat0.item()
            
        stats.append(stat - test_stat)
    return np.sort(stats)

def testing(model, z_test, x_test, y_test, n_res, device, dtype):
    pred = np.array(model.predict_proba(torch.cat((z_test, x_test, y_test))))
    if pred.ndim == 2:
        pred = pred[:, 1]
    N1 = len(z_test)
    
    z_pred = pred[:N1]
    x_pred = pred[N1:2*N1]
    y_pred = pred[2*N1:3*N1]
    
    Kzx = z_pred @ x_pred
    Kzy = z_pred @ y_pred
    Kxx = x_pred @ x_pred
    Kyy = y_pred @ y_pred
    
    hh = Kzx + Kzx.t() - Kzy - Kzy.t() - Kxx + Kyy

    test_stat = (torch.sum(hh)- hh.trace())/ (N1*(N1-1))

    stats = []
    for k in range(n_res):
        weights = np.random.exponential(scale=1, size = N1)
        weights = weights/np.mean(weights)
        hh_ = weights.t() @ weights * hh
        stat = (torch.sum(hh_)- hh_.trace())/ (N1*(N1-1))
        stat = stat.item()
            
        stats.append(stat - test_stat)
    # print(signal, p)
    return test_stat, stats

def TST_AUTO(name, N1, rs, check, n_test, n_res, alpha=0.05):
    np.random.seed(rs)
    X_train, Y_train = load_data(name, N1, rs, check)
    S_train = np.concatenate((X_train, Y_train), axis=0)
    label_train = np.concatenate(([1] * N1, [0] * N1))
    df_train = pd.DataFrame({"data"+str(i): S_train[:, i] for i in range(len(S_train[0]))})
    df_train["label"] = label_train
    train_data = TabularDataset(df_train)
    AutoML_predictor = TabularPredictor(label="label", problem_type="binary", eval_metric="accuracy", verbosity=0).fit(train_data, presets='best_quality', time_limit=60)

    H = np.zeros(n_test)
    P = np.zeros(n_test)
    
    N_test_all = 10 * N1
    X_test_all, Y_test_all = load_data(name, N_test_all, rs + 283, check)
    # test by C2ST-L
    for k in range(n_test):
        ind_test = np.random.choice(N_test_all, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]
        S_test = np.concatenate((X_test, Y_test), axis=0)
        label_test = np.concatenate(([1] * len(ind_test), [0] * len(ind_test)))
        df_test = pd.DataFrame({"data" + str(i): S_test[:, i] for i in range(len(S_test[0]))})
        df_test["label"] = label_test
        test_data = TabularDataset(df_test)
        test_stat, stats_res = testing(AutoML_predictor, test_data, label_test, n_res)
        thres = stats_res[int(n_res * (1-alpha))-1]
        H[k] = test_stat > thres
        P[k] = 1 - np.searchsorted(stats_res, test_stat, side="left")/n_res
    
    return H, P