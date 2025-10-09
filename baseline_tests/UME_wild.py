import numpy as np
import torch
from sklearn.utils import check_random_state
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader import load_data
def crit(val, var):
    std_temp = torch.sqrt(var+10**(-9)) #this is std
    return -torch.div(val, std_temp)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist_m(x, y, M_matrix):
    """compute the paired distance between x and y."""
    nx = x.shape[0]
    ny = y.shape[0]
    d = x.shape[1]
    
    x_Mat = torch.matmul(x, M_matrix)
    y_Mat = torch.matmul(y, M_matrix)    
    
    x_Mat_x = torch.sum(torch.mul(x_Mat, x), 1).view(-1, 1)
    y_Mat_y = torch.sum(torch.mul(y_Mat, y), 1).view(1, -1)
    x_Mat_y = torch.sum(torch.mul(x_Mat.unsqueeze(1).expand(nx,ny,d), y), 2)
    y_Mat_x = torch.sum(torch.mul(y_Mat.unsqueeze(1).expand(ny,nx,d), x), 2).t()
    
    Pdist = x_Mat_x + y_Mat_y - x_Mat_y - y_Mat_x
    Pdist[Pdist<0]=0
    return Pdist


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

def compute_UME_mean(Fea_X, X_org, Fea_Y, Y_org, Fea_V, V_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0):
        """
        Return the mean of the reduced
        Here we don't assume X.shape == Y.shape !
        X.shape = n1 x d, Y.shape = n2 x d, V.shape = J x d
        Return a scalar
        !!!! current version is biased
        """
        def compute_gram_matrix(Fea_X, X_org, Fea_Y, Y_org, sigma, sigma0=0.1, epsilon=10 ** (-10), cst = 1.0): 
                """
                Parameters
                ----------
                XY : (n1+n2) x d numpy array
                Return
                ------
                fea_pq : n1 x n2 numpy array
                """
                L = 1 # generalized Gaussian (if L>1)
                Dxy = Pdist2(Fea_X, Fea_Y)
                Dxy_org = Pdist2(X_org, Y_org)
                Kxy = cst*((1-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma))**L + epsilon * torch.exp(-Dxy_org / sigma))
                return Kxy
        
        Kxv = compute_gram_matrix(Fea_X, X_org, Fea_V, V_org, sigma, sigma0, epsilon, cst) # n1 x J
        Kyv = compute_gram_matrix(Fea_Y, Y_org, Fea_V, V_org, sigma, sigma0, epsilon, cst) # n2 x J

        fea_pq = 1/np.sqrt(len(Fea_V)) * (Kxv - Kyv)
        
        mu_p_V = torch.mean(Kxv, axis=0) # J vector
        mu_q_V = torch.mean(Kyv, axis=0) # J vector
        t1 = torch.mean((mu_p_V-mu_q_V)**2)
        t2 = 0
        UME_mean = t1 - t2

        # compute the variance
        mu = torch.mean(fea_pq, axis=0, keepdim=True) # J*1 vector
        mu = mu.t()
        # ! note that torch.dot does not support broadcasting
        UME_variance = 4.0*torch.mean(torch.matmul(fea_pq, mu)**2) - 4.0*torch.sum(mu**2)**2

        return UME_mean, UME_variance, Kxv, Kyv

def train_d(name, N1, V, X, Y, learning_rate, N_epoch, batch_size, device, dtype):
    batches=N1//batch_size
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
    for _ in range(N_epoch):
        indices = torch.randperm(N1)
        X = X[indices]
        indices = torch.randperm(N1)
        Y = Y[indices]
        total_S=[(X[i*batch_size:i*batch_size+batch_size], 
                    Y[i*batch_size:i*batch_size+batch_size]) for i in range(batches)]
        total_S=[torch.cat((X, Y)) for (X, Y) in total_S] #has shape batches x 2n x d
        for ind in range(batches):
            ep = torch.exp(epsilonOPT)/(1+torch.exp(epsilonOPT))
            sigma = sigmaOPT ** 2
            sigma0_u = sigma0OPT ** 2
            S=total_S[ind]
            modelu_output = model_u(S) 
            V_output = model_u(V)
            if len(S)//2 < batch_size:
                continue
            mmd_val, mmd_var, Kxv, Kyv = compute_UME_mean(modelu_output[:len(S)//2], S[:len(S)//2], modelu_output[len(S)//2:], S[len(S)//2:], V_output, V, sigma, sigma0_u, ep, cst)
            
            STAT_u = crit(mmd_val, mmd_var)
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
    model_u, ep, sigma, sigma0, V = train_d(name, N1, V, X_train, Y_train, learning_rate, N_epoch, batch_size, device, dtype)

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

            mmd_XZ, var_XZ, Kxv, Kzv = compute_UME_mean(Fea_X, X_test, Fea_Z, Z_test, Fea_V, V, sigma, sigma0, ep, cst=1.0)
            mmd_YZ, var_YZ, Kyv, Kzv = compute_UME_mean(Fea_Y, Y_test, Fea_Z, Z_test, Fea_V, V, sigma, sigma0, ep, cst=1.0)
            test_stat = mmd_XZ - mmd_YZ
            
            stats = []
            for j in range(n_res):
                weights = np.random.exponential(scale=1, size = N1)
                weights = torch.tensor(weights/np.mean(weights), device=device, dtype=dtype).reshape(1,-1)
                w_M = weights.t() @ weights
                stat = 0
                for i in range(len(V)):
                    hh1 = Kzv[:,i].reshape(-1,1) @ Kzv[:,i].reshape(1,-1) + Kxv[:,i].reshape(-1,1) @ Kxv[:,i].reshape(1,-1) - Kzv[:,i].reshape(-1,1) @ Kxv[:,i].reshape(1,-1) - Kxv[:,i].reshape(-1,1) @ Kzv[:,i].reshape(1,-1)
                    hh2 = Kzv[:,i].reshape(-1,1) @ Kzv[:,i].reshape(1,-1) + Kyv[:,i].reshape(-1,1) @ Kyv[:,i].reshape(1,-1) - Kzv[:,i].reshape(-1,1) @ Kyv[:,i].reshape(1,-1) - Kyv[:,i].reshape(-1,1) @ Kzv[:,i].reshape(1,-1)
                    hh = hh1-hh2
                    hh_ = w_M * hh
                    stat += (torch.sum(hh_))/ (len(Kzv)*(len(Kzv)))/len(V)
                stats.append(stat.item() - test_stat.item())
            stats = np.sort(stats)
            thres = stats[int(n_res * (1-alpha))-1]
            
            H[k] = test_stat.item() > thres
            P[k] = 1 - np.searchsorted(stats, test_stat.item(), side="left")/n_res
        
    return H, P
