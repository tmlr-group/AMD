import numpy as np
import torch
from sklearn.utils import check_random_state
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader import load_data
from baseline_tests.utils import *

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

class ConvNet_MNIST(nn.Module):
    def __init__(self):
        super(ConvNet_MNIST, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0)] #0.25
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            nn.Unflatten(1,(1,32,32)),
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 100))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        feature = self.adv_layer(out)

        return feature

class ModelLatentF(torch.nn.Module):
    """Latent space for both domains."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False
        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant




def crit(mmd_val, mmd_var, liuetal=True, Sharpe=False):
    if liuetal:
        mmd_std_temp = torch.sqrt(mmd_var+10**(-8)) #this is std
        return -1 * torch.div(mmd_val, mmd_std_temp)
    elif Sharpe:
        return mmd_val - 2.0 * mmd_var


def wildbootstrap(test_stat, Kxz, Kyz, n_res, device, dtype):
    N1 = len(Kxz)
    stats = []
    for k in range(n_res):
        weights = np.random.exponential(scale=1, size = N1)
        weights = torch.tensor(weights/np.mean(weights), device=device, dtype=dtype).reshape(1,-1)
        hh1_ = weights.t() @ weights * (Kxz - Kyz)
        stat1 = torch.div((torch.sum(hh1_)), (N1 * N1))
        stat = stat1.item()
            
        stats.append(stat - test_stat)
    return np.sort(stats)

def mmdGT(X, Y, model_u, n, sigma, sigma0, device, dtype, ep):
    S = torch.cat((X, Y), dim=0)
    Fea = model_u(S)
    n = X.shape[0]
    return MMD_LFI(Fea, n, S, sigma, sigma0, ep)

def train_d(name,n, X, Y, learning_rate, N_epoch, batch_size, device, dtype):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    
    if name in ["CIFAR10_ddpm","CIFAR10"]:
        model_u = ConvNet_CIFAR10().cuda()
        epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.array([10000.0]), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.array([0.1]), device, dtype)
        sigma0OPT.requires_grad = True
    elif name in ["MNIST"]:
        model_u = ConvNet_MNIST().cuda()
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*32*32), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = True
    elif name in ["BLOB"]:
        model_u = ModelLatentF(2, 50, 50).cuda()
        epsilonOPT = MatConvert(np.random.rand(1) * (10 ** (-10)), device, dtype)
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.sqrt(np.random.rand(1) * 0.3), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.sqrt(np.random.rand(1) * 0.002), device, dtype)
        sigma0OPT.requires_grad = True
    elif name in ["HDGM"]:
        model_u = ModelLatentF(10, 30, 30).cuda()
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 10), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.1), device, dtype)
        sigma0OPT.requires_grad = True
    elif name in ["HIGGS"]:
        model_u = ModelLatentF(4, 20, 20).cuda()
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
        epsilonOPT.requires_grad = True
        sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2*4), device, dtype)
        sigmaOPT.requires_grad = True
        sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), device, dtype)
        sigma0OPT.requires_grad = False
        
    cst=MatConvert(np.ones((1,)), device, dtype) # set to 1 to meet liu etal objective
    optimizer_u = torch.optim.Adam(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate)
    for t in range(N_epoch):
        batches=n//batch_size
        np.random.shuffle(X)
        np.random.shuffle(Y)
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[MatConvert(np.concatenate((X, Y), axis=0), device, dtype) for (X, Y) in total_S]
        for ind in range(batches):
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            S=total_S[ind]
            if len(S)//2 < batch_size:
                continue
            modelu_output = model_u(S)
            TEMP = MMDu(modelu_output, batch_size, S, sigma, sigma0_u, ep, cst)
            mmd_val = TEMP[0]
            mmd_var = TEMP[1]
            STAT_u = crit(mmd_val, mmd_var) 
            optimizer_u.zero_grad()
            STAT_u.backward(retain_graph=True)
            optimizer_u.step()
    return model_u, torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)), sigmaOPT ** 2, sigma0OPT ** 2

def KLFI_method(name, N1, rs, n_test, n_res, per, alpha, scale_coe, device, dtype, learning_rate, N_epoch, batch_size):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)

    X_train, Y_train, Z_train = load_data(name, int(N1*scale_coe), rs, 0.5)
    Z_train = MatConvert(Z_train, device, dtype)
    model_u, ep, sigma, sigma0=train_d(name, int(N1*scale_coe), X_train, Y_train, learning_rate, N_epoch, batch_size, device, dtype)
    with torch.no_grad():

        H = np.zeros(n_test)
        P = np.zeros(n_test)
        
        X_test_all, Y_test_all, Z_test_all = load_data(name, int(N1*scale_coe) * 10, rs + 2, per)
        X_test_all = MatConvert(X_test_all, device, dtype)
        Y_test_all = MatConvert(Y_test_all, device, dtype)
        Z_test_all = MatConvert(Z_test_all, device, dtype)
        for k in range(n_test):
            ind_test = np.random.choice(int(N1*scale_coe) * 10, int(N1*scale_coe), replace=True)
            X_test = X_test_all[ind_test]
            Y_test = Y_test_all[ind_test]
            Z_test = Z_test_all[ind_test]

            mmd_XZ, _ = mmdGT(X_test, Z_test, model_u, int(N1*scale_coe), sigma, sigma0, device, dtype, ep)
            mmd_YZ, _ = mmdGT(Y_test, Z_test, model_u, int(N1*scale_coe), sigma, sigma0, device, dtype, ep)

            test_value = float(mmd_YZ - mmd_XZ)
            
            ind_res = np.random.choice(int(N1*scale_coe), int(N1*scale_coe)//2, replace=True)
            Z_res = torch.cat((X_test[ind_res],Y_test[ind_res]))
            
            _, Kxz = mmdGT(X_test, Z_res, model_u, int(N1*scale_coe), sigma, sigma0, device, dtype, ep)
            _, Kyz = mmdGT(Y_test, Z_res, model_u, int(N1*scale_coe), sigma, sigma0, device, dtype, ep)
            stats_res = wildbootstrap(0, Kyz, Kxz, n_res, device, dtype)
            thres = stats_res[int(n_res * (1-alpha))-1]
            
            H[k] = test_value > thres
            P[k] = 1 - np.searchsorted(stats_res, test_value, side="left")/n_res

    return H, P