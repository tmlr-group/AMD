import numpy as np
import torch
import torch.nn as nn
import os
import sys

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_DIR not in sys.path:
    sys.path.append(_REPO_DIR)
from dataloader import load_data

_AMD_CORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _AMD_CORE_DIR not in sys.path:
    sys.path.append(_AMD_CORE_DIR)
from amd_core import select_rst_direction_grid

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

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

def Pdist2(x, y, M_matrix):
    """compute the paired distance between x and y."""
    n = x.shape[0]
    d = x.shape[1]
    assert x.shape[0]==y.shape[0]
    
    x_Mat = torch.matmul(x, M_matrix)
    y_Mat = torch.matmul(y, M_matrix)    
    
    x_Mat_x = torch.sum(torch.mul(x_Mat, x), 1).view(-1, 1)
    y_Mat_y = torch.sum(torch.mul(y_Mat, y), 1).view(1, -1)
    x_Mat_y = torch.sum(torch.mul(x_Mat.unsqueeze(1).expand(n,n,d), y), 2)
    y_Mat_x = torch.sum(torch.mul(y_Mat.unsqueeze(1).expand(n,n,d), x), 2).t()
    
    Pdist = x_Mat_x + y_Mat_y - x_Mat_y - y_Mat_x
    Pdist[Pdist<0]=0
    return Pdist

def PdistL(x, y):
    """compute the paired distance between x and y."""
    n = x.shape[0]
    d = x.shape[1]
    assert x.shape[0] == y.shape[0]
    x = x.unsqueeze(1)
    x = x.expand(n,n,d)
    Pdist = abs(x-y).sum(axis=2)
    return Pdist * np.sqrt(2)

def Ustats_RST(Fea, len_s, Fea_org, sigma, sigma0, epsilon, cst, is_smooth = True, M_matrix = None, kernel='Gaussian'):
    X = Fea[0:len_s, :]
    Y = Fea[len_s:2*len_s, :]
    Z = Fea[2*len_s:, :]
    
    if is_smooth:
        assert kernel == 'Deep'
        Dxx = Pdist(X, X)
        Dyy = Pdist(Y, Y)
        Dzx = Pdist(Z, X)
        Dzy = Pdist(Z, Y)
    
        X_org = Fea_org[0:len_s, :]
        Y_org = Fea_org[len_s:2*len_s, :]
        Z_org = Fea_org[2*len_s:, :]
        Dxx_org = Pdist(X_org, X_org)
        Dyy_org = Pdist(Y_org, Y_org)
        Dzx_org = Pdist(Z_org, X_org)
        Dzy_org = Pdist(Z_org, Y_org)
        
        L = 1 # generalized Gaussian (if L>1)
        Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
        Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
        Kzx = cst*((1-epsilon) * torch.exp(-(Dzx / sigma0) - (Dzx_org / sigma))**L + epsilon * torch.exp(-Dzx_org / sigma))
        Kzy = cst*((1-epsilon) * torch.exp(-(Dzy / sigma0) - (Dzy_org / sigma))**L + epsilon * torch.exp(-Dzy_org / sigma))

    else:
        if kernel == 'Gaussian':
            Dxx = Pdist(X, X)
            Dyy = Pdist(Y, Y)
            Dzx = Pdist(Z, X)
            Dzy = Pdist(Z, Y)
        elif kernel == "Laplace":
            Dxx = PdistL(X, X)
            Dyy = PdistL(Y, Y)
            Dzx = PdistL(Z, X)
            Dzy = PdistL(Z, Y)
        elif kernel =='Mahalanobis':
            Dxx = Pdist2(X, X, M_matrix)
            Dyy = Pdist2(Y, Y, M_matrix)
            Dzx = Pdist2(Z, X, M_matrix)
            Dzy = Pdist2(Z, Y, M_matrix)

        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kzx = torch.exp(-Dzx / sigma0)
        Kzy = torch.exp(-Dzy / sigma0)

    hh = (-Kzx - Kzx.t() + Kzy + Kzy.t() + Kx - Ky)/2
    
    stat = (torch.sum(hh)- hh.trace())/ (len_s*(len_s-1))

    return stat, hh

def Ustats_TST(Fea, len_s, Fea_org, sigma, sigma0, epsilon, cst, is_smooth = True, M_matrix = None, kernel='Gaussian'):
    X = Fea[0:len_s, :]
    Y = Fea[len_s:2*len_s, :]
    
    if is_smooth:
        assert kernel == 'Deep'
        Dxx = Pdist(X, X)
        Dyy = Pdist(Y, Y)
        Dxy = Pdist(X, Y)
        Dyx = Pdist(Y, X)
    
        X_org = Fea_org[0:len_s, :]
        Y_org = Fea_org[len_s:2*len_s, :]
        Dxx_org = Pdist(X_org, X_org)
        Dyy_org = Pdist(Y_org, Y_org)
        Dxy_org  = Pdist(X_org , Y_org )
        Dyx_org  = Pdist(Y_org , X_org )
        
        L = 1 # generalized Gaussian (if L>1)
        Kx = cst*((1-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma))**L + epsilon * torch.exp(-Dxx_org / sigma))
        Ky = cst*((1-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma))**L + epsilon * torch.exp(-Dyy_org / sigma))
        Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
        Kyx = cst*((1-epsilon) * torch.exp(-(Dyx / sigma0) - (Dyx_org / sigma))**L + epsilon * torch.exp(-Dyx_org / sigma))

    else:
        if kernel == 'Gaussian':
            Dxx = Pdist(X, X)
            Dyy = Pdist(Y, Y)
            Dxy = Pdist(X, Y)
            Dyx = Pdist(Y, X)
        elif kernel == "Laplace":
            Dxx = PdistL(X, X)
            Dyy = PdistL(Y, Y)
            Dxy = PdistL(X, Y)
            Dyx = PdistL(Y, X)
        elif kernel =='Mahalanobis':
            Dxx = Pdist2(X, X, M_matrix)
            Dyy = Pdist2(Y, Y, M_matrix)
            Dxy = Pdist2(X, Y, M_matrix)
            Dyx = Pdist2(Y, X, M_matrix)

        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
        Kyx = torch.exp(-Dyx / sigma0)

    hh = Kx+Ky-Kxy-Kyx
    
    stat = (torch.sum(hh)- hh.trace())/ (len_s*(len_s-1))

    return stat, hh

def training(kernel, N1, X, Y, Z, C, learning_rate, N_epoch, batch_size, device, dtype, F, heur):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = True
    
    M_matrix = np.identity(X.shape[1])
    M_matrix = MatConvert(M_matrix, device, dtype)
    M_matrix.requires_grad = True
    
    if kernel == 'Gaussian':
        Dxy = Pdist(X, Y)
        Dzx = Pdist(Z, X)
        Dzy = Pdist(Z, Y)
    elif kernel == "Laplace":
        Dxy = PdistL(X, Y)
        Dzx = PdistL(Z, X)
        Dzy = PdistL(Z, Y)
    elif kernel =='Mahalanobis':
        Dxy = Pdist2(X, Y, M_matrix)
        Dzx = Pdist2(Z, X, M_matrix)
        Dzy = Pdist2(Z, Y, M_matrix)

    if heur == 'TST':
        sigma0OPT = Dxy.median()
    elif heur == 'RST':
        sigma_zx=torch.sqrt(Dzx.median()/2)
        sigma_zy=torch.sqrt(Dzy.median()/2)
        sigma0OPT = (sigma_zx+sigma_zy)/2
        
    sigma0OPT.requires_grad = True
    optimizer = torch.optim.RMSprop([sigma0OPT] + [M_matrix], lr=learning_rate, alpha=0.9 ,momentum=0.9, eps=1e-7)
    batches=N1//batch_size
    for _ in range(N_epoch):
        indices = torch.randperm(N1)
        X1 = X[indices]
        indices = torch.randperm(N1)
        Y1 = Y[indices]
        indices = torch.randperm(N1)
        Z1 = Z[indices]
        total_S=[(X1[i*batch_size:i*batch_size+batch_size], 
                    Y1[i*batch_size:i*batch_size+batch_size], 
                    Z1[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[torch.cat((X, Y, Z)) for (X, Y, Z) in total_S]
        if F is not None:
            indicesX = np.random.choice(np.arange(0,N1), N1)
            indicesY = np.random.choice(np.arange(0,N1), N1)
            X_ = 0.5 * X[indicesX] + 0.5 * Y[indicesY]
            indicesX = np.random.choice(np.arange(0,N1), N1)
            indicesY = np.random.choice(np.arange(0,N1), N1)
            Y_ = 0.5 * X[indicesX] + 0.5 * Y[indicesY]
            indices = np.random.choice(np.arange(0,N1), N1)
            Z_ = Z[indices]
            total_S_=[(X_[i*batch_size:i*batch_size+batch_size], 
                        Y_[i*batch_size:i*batch_size+batch_size], 
                        Z_[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
            total_S_=[torch.cat((X_, Y_, Z_)) for (X_, Y_, Z_) in total_S_]
        
        for idx in range(batches):
            sigma0 = sigma0OPT ** 2
            S = total_S[idx]
            if len(S)//3 < batch_size:
                continue

            if F is not None:
                TEMP = Ustats_RST(S, len(S)//3, None, None, sigma0, None, None, is_smooth = False, M_matrix = M_matrix, kernel = kernel)
                S_ = total_S_[idx]
                TEMP_ = Ustats_RST(S_, len(S_)//3, None, None, sigma0, None, None, is_smooth = False, M_matrix = M_matrix, kernel = kernel)
                opt_value = -F * TEMP[0] + C * TEMP_[0]**2
            else:
                TEMP = Ustats_TST(S, len(S)//3, None, None, sigma0, None, None, is_smooth = False, M_matrix = M_matrix, kernel = kernel)
                opt_value = -1 * TEMP[0]
                
            optimizer.zero_grad()
            opt_value.backward(retain_graph=True)
            optimizer.step()
            
    with torch.no_grad(): 
        sigma0 = sigma0OPT ** 2
        TEMP_ALL = Ustats_RST(torch.cat((X, Y, Z)), N1, None, None, sigma0, None, None, is_smooth = False, M_matrix = M_matrix, kernel = kernel)
        return TEMP_ALL[0].item(), [sigma0.detach(), M_matrix.detach()]

def training_Deep(name, N1, X, Y, Z, C, learning_rate, N_epoch, batch_size, device, dtype, F, heur):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    torch.backends.cudnn.deterministic = True
    cst=MatConvert(np.ones((1,)), device, dtype)
    
    if name in ["CIFAR10_ddpm"]:
        model_u = ConvNet_CIFAR10().cuda()
        epsilonOPT = MatConvert(np.array([-1.0]), device, dtype)
    elif name in ["MNIST"]:
        model_u = ConvNet_MNIST().cuda()
        epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), device, dtype))
    else:
        print('Need to specify a model')
    
    Dxy = Pdist(X, Y)
    Dzx = Pdist(Z, X)
    Dzy = Pdist(Z, Y)
    
    with torch.no_grad():
        Fea_X = model_u(X)
        Fea_Y = model_u(Y)
        Fea_Z = model_u(Z)
        Fea_Dxy = Pdist(Fea_X, Fea_Y)
        Fea_Dzx = Pdist(Fea_Z, Fea_X)
        Fea_Dzy = Pdist(Fea_Z, Fea_Y)
    
    if heur == 'TST':
        sigmaOPT = Dxy.median()
        sigma0OPT = Fea_Dxy.median().detach()
    elif heur == 'RST':
        sigma_zx=torch.sqrt(Dzx.median()/2)
        sigma_zy=torch.sqrt(Dzy.median()/2)
        sigmaOPT = (sigma_zx+sigma_zy)/2
        sigma0_zx=torch.sqrt(Fea_Dzx.median()/2)
        sigma0_zy=torch.sqrt(Fea_Dzy.median()/2)
        sigma0OPT = (sigma0_zx+sigma0_zy)/2
        
    epsilonOPT.requires_grad = True
    sigmaOPT.requires_grad = True
    sigma0OPT.requires_grad = True
    optimizer = torch.optim.RMSprop(list(model_u.parameters())+[epsilonOPT]+[sigmaOPT]+[sigma0OPT], lr=learning_rate, alpha=0.9 ,momentum=0.9, eps=1e-7)
    batches=N1//batch_size
    for _ in range(N_epoch):
        indices = torch.randperm(N1)
        X1 = X[indices]
        indices = torch.randperm(N1)
        Y1 = Y[indices]
        indices = torch.randperm(N1)
        Z1 = Z[indices]
        total_S=[(X1[i*batch_size:i*batch_size+batch_size], 
                    Y1[i*batch_size:i*batch_size+batch_size], 
                    Z1[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[torch.cat((X, Y, Z)) for (X, Y, Z) in total_S]
 
        if F is not None:
            indicesX = np.random.choice(np.arange(0,N1), N1)
            indicesY = np.random.choice(np.arange(0,N1), N1)
            X_ = 0.5 * X[indicesX] + 0.5 * Y[indicesY]
            indicesX = np.random.choice(np.arange(0,N1), N1)
            indicesY = np.random.choice(np.arange(0,N1), N1)
            Y_ = 0.5 * X[indicesX] + 0.5 * Y[indicesY]
            indices = np.random.choice(np.arange(0,N1), N1)
            Z_ = Z[indices]
            total_S_=[(X_[i*batch_size:i*batch_size+batch_size], 
                        Y_[i*batch_size:i*batch_size+batch_size], 
                        Z_[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
            total_S_=[torch.cat((X_, Y_, Z_)) for (X_, Y_, Z_) in total_S_]
            
        for idx in range(batches):
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0 = sigma0OPT ** 2
            S = total_S[idx]
            if len(S)//3 < batch_size:
                continue
            Fea_S = model_u(S)
            if F is not None:
                TEMP = Ustats_RST(Fea_S, len(S)//3, S, sigma, sigma0, ep, cst, kernel="Deep")
                S_ = total_S_[idx]
                Fea_S_ = model_u(S_)
                TEMP_ = Ustats_RST(Fea_S_, len(S_)//3, S_, sigma, sigma0, ep, cst, kernel="Deep")
                opt_value = -F * TEMP[0] + C * TEMP_[0]**2
            else:
                TEMP = Ustats_TST(Fea_S, len(S)//3, S, sigma, sigma0, ep, cst, kernel="Deep")
                opt_value = -1* TEMP[0]
            optimizer.zero_grad()
            opt_value.backward(retain_graph=True)
            optimizer.step()
    with torch.no_grad(): 
        ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
        sigma = sigmaOPT ** 2
        sigma0 = sigma0OPT ** 2
        TEMP_ALL = Ustats_RST(model_u(torch.cat((X, Y, Z))), N1, torch.cat((X, Y, Z)), sigma, sigma0, ep, cst, kernel="Deep")
        return TEMP_ALL[0].item(), [model_u, torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT)).detach(), sigmaOPT.detach() ** 2, sigma0OPT.detach() ** 2, cst]

def F_selection(name, F_kernel, N1, X, Y, Z, C, learning_rate, N_epoch, batch_size, device, dtype, F_select='RST', F_heur ='TST'):
    if F_select == 'Random':
        F = np.random.choice([1, -1])
    else:
        F, _ = select_rst_direction_grid(
            X,
            Y,
            Z,
            kernel=F_kernel,
            seed=1102 + int(N1),
            max_samples=min(int(N1), 2048),
            num_bandwidths=15,
            num_bags=3,
            top_k=7,
        )
    return int(F)

def wildbootstrapWD2(test_stat, hh1, n_res, device, dtype, F):
    N1 = len(hh1)
    stats = []
    for k in range(n_res):
        weights = np.random.exponential(scale=1, size = N1)
        weights = torch.tensor(weights/np.mean(weights), device=device, dtype=dtype).reshape(1,-1)
        hh1_ = weights.t() @ weights * hh1
        stat1 = (torch.sum(hh1_)- hh1_.trace())/ (N1*(N1-1))
        stat = stat1.item()
        stats.append(F*(stat-test_stat))
        
    return np.sort(stats)

def Ours_method(name, N1, rs, n_test, n_res, per, alpha, device, dtype, kernel, heur, learning_rate, N_epoch, C, batch_size, F_kernel="Laplace", F_select="RST", F_heur="RST", learning_rate_F=0.1, N_epoch_F=300, C_F=5, batch_size_F=32):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    phase1_n = max(int(N1), 2048)
    X_phase1, Y_phase1, Z_phase1 = load_data(name, phase1_n, rs + 7919, per)
    X_phase1 = MatConvert(X_phase1, device, dtype)
    Y_phase1 = MatConvert(Y_phase1, device, dtype)
    Z_phase1 = MatConvert(Z_phase1, device, dtype)
    F = F_selection(name, F_kernel, phase1_n, X_phase1, Y_phase1, Z_phase1, C_F, learning_rate_F, N_epoch_F, min(batch_size_F, phase1_n), device, dtype, F_select, F_heur)

    X_train, Y_train, Z_train = load_data(name, N1, rs, per)
    X_train = MatConvert(X_train, device, dtype)
    Y_train = MatConvert(Y_train, device, dtype)
    Z_train = MatConvert(Z_train, device, dtype)

    if kernel == 'Deep':
        val, TEMP = training_Deep(name, N1, X_train, Y_train, Z_train, C, learning_rate, N_epoch, batch_size, device, dtype, F, heur)
        model, ep, sigma, sigma0, cst = TEMP
    else:
        val, TEMP= training(kernel, N1, X_train, Y_train, Z_train, C, learning_rate, N_epoch, batch_size, device, dtype, F, F_heur)
        sigma0, M_matrix = TEMP
    
    H = np.zeros(n_test)
    P = np.zeros(n_test)
    H2 = np.zeros(n_test)
    P2 = np.zeros(n_test)
    X_test_all, Y_test_all, Z_test_all = load_data(name, N1 * 10, rs + 2, per)
    X_test_all = MatConvert(X_test_all, device, dtype)
    Y_test_all = MatConvert(Y_test_all, device, dtype)
    Z_test_all = MatConvert(Z_test_all, device, dtype)
    for k in range(n_test):
        ind_test = np.random.choice(N1*10, N1, replace=False)
        X_test = X_test_all[ind_test]
        Y_test = Y_test_all[ind_test]
        Z_test = Z_test_all[ind_test]
        
        S = torch.cat((X_test, Y_test, Z_test), dim=0)
        if kernel == 'Deep':
            Fea_S = model(S)
            TEMP = Ustats_RST(Fea_S, N1, S, sigma, sigma0, ep, cst, kernel = kernel)
        else:
            TEMP = Ustats_RST(S, N1, None, None, sigma0, None, None, is_smooth = False, M_matrix = M_matrix, kernel = kernel)
        test_stat = TEMP[0].item()

        stats_res = wildbootstrapWD2(test_stat, TEMP[1], n_res, device, dtype, F)
        thres = stats_res[int(n_res * (1-alpha))-1]
        H[k] = F * test_stat > thres
        P[k] = 1 - np.searchsorted(stats_res, F * test_stat, side="left")/n_res
        
        stats_res_abs = np.sort(np.abs(stats_res))
        thres_abs = stats_res_abs[int(n_res * (1-alpha))-1]
        H2[k] = abs(test_stat) > thres_abs
        P2[k] = 1 - np.searchsorted(stats_res_abs, abs(test_stat), side="left")/n_res
        
    return H, P, H2, P2
