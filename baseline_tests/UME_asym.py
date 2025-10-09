import numpy as np
import torch
from sklearn.utils import check_random_state
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader import load_data
from .utils import *
import scipy as sp
def crit(val, var):
    std_temp = torch.sqrt(var+10**(-9)) #this is std
    return -torch.div(val, std_temp)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

class ConvNet_CIFAR10(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR10, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            block =([nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.2, inplace=True),  
                     nn.Dropout2d(0)])
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        self.model = nn.Sequential(
            nn.Unflatten(1,(3,32,32)),
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        ds_size = 2
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 300))
    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)
        return feature

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    ###Supports m times n operation where m is not n
    ###takes input with shape (n, out) and (m, out), returns output with shape (n, m)
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def train_d(N1, V, X, Y, Z, learning_rate, N_epoch, batch_size, device, dtype):
    batches=N1//batch_size
    model_u = ConvNet_CIFAR10().cuda()
    epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
    epsilonOPT.requires_grad = True
    sigmaOPT = MatConvert(np.array([10000.0]), device, dtype)
    sigmaOPT.requires_grad = True
    sigma0OPT = MatConvert(np.array([0.1]), device, dtype)
    sigma0OPT.requires_grad = True
    cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
    optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT]+[V], lr=learning_rate)
    for _ in range(N_epoch):
        indices = torch.randperm(N1)
        X = X[indices]
        indices = torch.randperm(N1)
        Y = Y[indices]
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size], 
                    Z[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[torch.cat((X, Y, Z)) for (X, Y, Z) in total_S] #has shape batches x 2n x d
        for ind in range(batches):
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            S=total_S[ind]
            modelu_output = model_u(S) 
            V_output = model_u(V)
            if len(S)//3 < batch_size:
                continue
            
            fea_XZ = compute_feature_matrix(V_output, V, torch.cat((modelu_output[:batch_size],modelu_output[2*batch_size:])), torch.cat((S[:batch_size],S[2*batch_size:])), batch_size, sigma, sigma0_u, ep, cst)
            mmd_XZ, mmd_var_XZ = compute_UME_mean_variance(fea_XZ, batch_size)
            
            fea_YZ = compute_feature_matrix(V_output, V, torch.cat((modelu_output[batch_size:2*batch_size],modelu_output[2*batch_size:])), torch.cat((S[batch_size:2*batch_size],S[2*batch_size:])), batch_size, sigma, sigma0_u, ep, cst)
            mmd_YZ, mmd_var_YZ = compute_UME_mean_variance(fea_YZ, batch_size)
            
            mean_h1 = mmd_XZ - mmd_YZ
            t1 = 4.0*torch.mean(torch.matmul(fea_XZ, torch.mean(fea_XZ, dim=0))*torch.matmul(fea_YZ, fea_YZ.T))
            t2 = 4.0*torch.sum(torch.mean(fea_XZ, dim=0)**2)*torch.sum(torch.mean(fea_YZ, dim=0)**2)
            
            var_pqr = t1 - t2
            var_h1 = mmd_var_XZ - 2.0*var_pqr + mmd_var_YZ
            
            STAT_u = crit(mean_h1, var_h1)
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            optimizer_u.step()
    return model_u, torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)), sigmaOPT ** 2, sigma0OPT ** 2, V

def UME_method(name, N1, rs, v_num, n_test, n_res, per, alpha, device, dtype, learning_rate, N_epoch, batch_size):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    V, _, _ = load_data(name, v_num, rs, per)
    V = MatConvert(V, device, dtype)
    X_train, Y_train, Z_train = load_data(name, N1, rs, per)
    X_train = MatConvert(X_train, device, dtype)
    Y_train = MatConvert(Y_train, device, dtype)
    Z_train = MatConvert(Z_train, device, dtype)
    model_u, ep, sigma, sigma0, V = train_d(N1, V, X_train, Y_train, Z_train, learning_rate, N_epoch, batch_size, device, dtype)

    with torch.no_grad():
        H = np.zeros(n_test)
        P = np.zeros(n_test)
        
        X_test_all, Y_test_all, Z_test_all = load_data(name, N1 * 10, rs + 2, per)
        X_test_all = MatConvert(X_test_all, device, dtype)
        Y_test_all = MatConvert(Y_test_all, device, dtype)
        Z_test_all = MatConvert(Z_test_all, device, dtype)

        Fea_V=model_u(V)
        Fea_X_test_all = model_u(X_test_all)
        Fea_Y_test_all = model_u(Y_test_all)
        Fea_Z_test_all = model_u(Z_test_all)
        for k in range(n_test):
            ind_test = np.random.choice(N1*10, N1, replace=False)
            X_test = X_test_all[ind_test]
            Y_test = Y_test_all[ind_test]
            Z_test = Z_test_all[ind_test]
            Fea_X = Fea_X_test_all[ind_test]
            Fea_Y = Fea_Y_test_all[ind_test]
            Fea_Z = Fea_Z_test_all[ind_test]

            
            fea_XZ = compute_feature_matrix(Fea_V, V, torch.cat((Fea_X,Fea_Z)), torch.cat((X_test,Y_test)), N1, sigma, sigma0, ep)
            mmd_XZ, mmd_var_XZ = compute_UME_mean_variance(fea_XZ, N1)
            
            fea_YZ = compute_feature_matrix(Fea_V, V, torch.cat((Fea_Y,Fea_Z)), torch.cat((Y_test,Z_test)), N1, sigma, sigma0, ep)
            mmd_YZ, mmd_var_YZ = compute_UME_mean_variance(fea_YZ, N1)
            
            mean_h1 = mmd_XZ - mmd_YZ
            t1 = 4.0*torch.mean(torch.matmul(fea_XZ, torch.mean(fea_XZ, dim=0))*torch.matmul(fea_YZ, fea_YZ.T))
            t2 = 4.0*torch.sum(torch.mean(fea_XZ, dim=0)**2)*torch.sum(torch.mean(fea_YZ, dim=0)**2)
            
            var_pqr = t1 - t2
            var_h1 = mmd_var_XZ - 2.0*var_pqr + mmd_var_YZ
            
            STAT_u = crit(mean_h1, var_h1)

            pvalue=sp.stats.norm.cdf(STAT_u.item())
            
            H[k] = pvalue < alpha
            P[k] = pvalue
        
    return H, P