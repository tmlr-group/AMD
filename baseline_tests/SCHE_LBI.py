import numpy as np
import torch
from sklearn.utils import check_random_state
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath('..'))
from dataloader import load_data

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



def train(name, X, Y, criterion, batch_size, lr, epochs, device):
    """Label the items first
        items in X have label 0, items in Y have label 1
        then train the model with sgd
        X, Y are numpy arrays
        X has shape (N1, 2), Y has shape (N2, 2)
    """
    if name in ["CIFAR10_ddpm","CIFAR10"]:
        model = ConvNet_CIFAR10().cuda()
    elif name in ["MNIST"]:
        model = ConvNet_MNIST().cuda()
    elif name in ["BLOB"]:
        model = ModelLatentF(2, 50, 50).cuda()
    elif name in ["HDGM"]:
        model = ModelLatentF(10, 30, 30).cuda()
    elif name in ["HIGGS"]:
        model = ModelLatentF(4, 20, 20).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        indices = torch.randperm(len(X))
        X = X[indices]
        indices = torch.randperm(len(Y))
        Y = Y[indices]
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            output_X = model(X_batch)
            output_Y = model(Y_batch)
            loss_X = criterion(output_X, torch.zeros(len(X_batch)).long().to(device))
            loss_Y = criterion(output_Y, torch.ones(len(Y_batch)).long().to(device))
            loss = loss_X + loss_Y
            loss.backward()
            optimizer.step()
    return model

def Scheffe_LBI(model,Z):
    '''
    Output fraction of Z's classified as class 0
    '''
    with torch.no_grad():
        output_Z = model(Z)
        prob = torch.softmax(output_Z, dim=1)[:,1] #Probability of class 1
    
        class_ = torch.argmax(output_Z, dim=1)
        success = torch.sum(class_ == 1.0).float() / len(Z)
        return float(success), np.mean(prob.detach().cpu().numpy())

def SCHE_LBI_method(name, N1, rs, n_test, n_res, per, alpha, scale_coe, device, dtype, learning_rate, N_epoch, batch_size):
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    X_train, Y_train, _ = load_data(name, N1, rs, per)
    X_train = MatConvert(X_train, device, dtype)
    Y_train = MatConvert(Y_train, device, dtype)
    model = train(name, X_train, Y_train, nn.CrossEntropyLoss(), batch_size, learning_rate, N_epoch, device)
    
    H_Sche = np.zeros(n_test)
    P_Sche = np.zeros(n_test)
    H_LBI = np.zeros(n_test)
    P_LBI = np.zeros(n_test)

    with torch.no_grad():
        X_test_all, Y_test_all, Z_test_all = load_data(name, N1 * 10, rs + 2, per)
        X_test_all = MatConvert(X_test_all, device, dtype)
        Y_test_all = MatConvert(Y_test_all, device, dtype)
        Z_test_all = MatConvert(Z_test_all, device, dtype)
        for k in range(n_test):
            ind_test = np.random.choice(N1*10, N1, replace=False)
            X_test = X_test_all[ind_test]
            Y_test = Y_test_all[ind_test]
            Z_test = Z_test_all[ind_test]
            
            stats_Sche = []
            stats_LBI = []
            for j in range(int(n_res)):
                ind_res = np.random.choice(N1, N1//scale_coe, replace=True)
                Z_res = torch.cat((X_test[ind_res],Y_test[ind_res]))
                stat_SChe, stat_LBI = Scheffe_LBI(model, Z_res)
                stats_Sche.append(stat_SChe)
                stats_LBI.append(stat_LBI)
            
            Sche_test, LBI_test = Scheffe_LBI(model, Z_test)
            
            stats_Sche = np.sort(stats_Sche)
            thres_Sche = stats_Sche[int(n_res * (1-alpha))-1]
            H_Sche[k] = Sche_test > thres_Sche
            P_Sche[k] = 1 - np.searchsorted(stats_Sche, Sche_test, side="left")/n_res
            
            stats_LBI = np.sort(stats_LBI)
            thres_LBI = stats_LBI[int(n_res * (1-alpha))-1]
            H_LBI[k] = LBI_test > thres_LBI
            P_LBI[k] = 1 - np.searchsorted(stats_LBI, LBI_test, side="left")/n_res
            
        return H_Sche, P_Sche, H_LBI, P_LBI
