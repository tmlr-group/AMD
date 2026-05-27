import numpy as np
import torch
from sklearn.utils import check_random_state
from sklearn.preprocessing import MinMaxScaler
import pickle
import torchvision.transforms as transforms
from torchvision import datasets
import sys
import os
def MMscaler(X, Y, Z):
    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X, Y, Z), axis=0))
    return scaler.transform(X), scaler.transform(Y), scaler.transform(Z)

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def sample_cifar10_ddpm(N, rs, per, scale):
#     ##### data generation #####
#     n = 10000
#     try:
#         samples = np.load("ddpm_generated_images.npy")
#         assert len(samples) == n
#     except Exception as e:
#         model_id = "google/ddpm-cifar10-32"
#         ddpm = DDPMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
#         # number of samples 
#         for i in range(n):
#             print('{}-th sample'.format(i))
#             ddpm_output = ddpm()
#             image = np.asarray(ddpm_output.images[0], dtype=float)[np.newaxis, :, :, :]
#             # save images array
#             try:
#                 samples = np.load("ddpm_generated_images.npy")
#                 np.save("ddpm_generated_images.npy", np.concatenate((samples,image)))
#             except Exception as e:
#                 np.save("ddpm_generated_images.npy", image)
#             if len(samples) == n:
#                 break
#     ##### data generation #####
# sample_cifar10_ddpm(1, 1, 1, 1)

    np.random.seed(seed=rs)
    img_size = 32
    dataset_test = datasets.CIFAR10(root='cifar10', download=False, train=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        # transforms.Grayscale(),
                                    ]))

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                                  shuffle=True)
    # Obtain CIFAR10 images
    for i, (imgs, Labels) in enumerate(dataloader_test):
        data_all = np.array(imgs.view(len(imgs), -1))
    
    data_new = np.load("ddpm_generated_images.npy").transpose(0,3,1,2)
    data_T = data_new.reshape((-1, 3, img_size, img_size))
    ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
    data_T = data_T[ind_M]
    TT = transforms.Compose([transforms.Resize(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             transforms.Grayscale(),
                             ])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_T), 3, img_size, img_size])
    data_T_tensor = torch.from_numpy(data_T)
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        data_trans[i] = TT(d0)
    data_trans = np.array(data_trans.view(len(data_trans), -1))

    Ind = np.random.choice(len(data_all), N, replace=False)
    X = data_all[Ind]
    Ind_v4 = np.random.choice(len(data_trans), N, replace=False)
    Y = data_trans[Ind_v4]
    
    LenX = int(N * per)
    LenY = N-LenX
    # np.random.shuffle(Z)
    Ind_X = np.random.choice(len(data_all), LenX, replace=False)
    Ind_Y = np.random.choice(len(data_trans), LenY, replace=False)
    Z = np.concatenate((data_all[Ind_X],data_trans[Ind_Y]), axis=0)
    np.random.shuffle(Z)

    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def sample_BLOB(N, rs, per, scale):
    """#Feat. 2  # Inst. inf"""
    rs = check_random_state(rs)
    rows = 3
    cols = 3

    """Generate Blob-D for testing type-II error (or test power)"""
    sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
    sigma_mx_2 = np.zeros([9, 2, 2])
    for i in range(9):
        sigma_mx_2[i] = sigma_mx_2_standard
        if i < 4:
            sigma_mx_2[i][0, 1] = -0.02 - 0.002 * i
            sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
        if i == 4:
            sigma_mx_2[i][0, 1] = 0.00
            sigma_mx_2[i][1, 0] = 0.00
        if i > 4:
            sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i - 5)
            sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i - 5)

    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    X = rs.multivariate_normal(mu, sigma, size=N)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N)
    X[:, 1] += rs.randint(cols, size=N)

    Y = rs.multivariate_normal(mu, np.eye(2), size=N)
    Y_row = rs.randint(rows, size=N)
    Y_col = rs.randint(cols, size=N)
    locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y, L) + locs[i], Y)

    LenX = int(N * per)
    LenY = N-LenX
    ZX = rs.multivariate_normal(mu, sigma, size=LenX)
    # assign to blobs
    ZX[:, 0] += rs.randint(rows, size=LenX)
    ZX[:, 1] += rs.randint(cols, size=LenX)
    ZY = rs.multivariate_normal(mu, np.eye(2), size=LenY)
    ZY_row = rs.randint(rows, size=LenY)
    ZY_col = rs.randint(cols, size=LenY)
    locs = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((ZY_row == locs[i][0]) & (ZY_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        ZY = np.where(ind2, np.matmul(ZY, L) + locs[i], ZY)
    Z = np.concatenate((ZX,ZY),axis=0)
    np.random.shuffle(Z)
    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def sample_HDGM(N, rs, per, scale):
    """#Feat. 10  # Inst. inf"""
    d = 10 # data dim
    Num_clusters = 2  # number of modes
    def cluster_counts(total):
        counts = np.full(Num_clusters, total // Num_clusters, dtype=int)
        counts[: total % Num_clusters] += 1
        return counts

    mu_mx = np.zeros([Num_clusters, d])
    mu_mx[1] = mu_mx[1] + 0.5
    sigma_mx_1 = np.identity(d)
    X = np.zeros([N, d])
    Y = np.zeros([N, d])
    # Generate HDGM-D
    sample_counts = cluster_counts(N)
    start = 0
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i + 283)
        count = sample_counts[i]
        X[start:start + count, :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, count)
        start += count
    start = 0
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i)

        sigma_mx_2 = [np.identity(d), np.identity(d)]
        sigma_mx_2[0][0, 1] = 0.5
        sigma_mx_2[0][1, 0] = 0.5
        sigma_mx_2[1][0, 1] = -0.5
        sigma_mx_2[1][1, 0] = -0.5
        count = sample_counts[i]
        Y[start:start + count, :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], count)
        start += count
        
    LenX = int(N * per)
    LenY = N-LenX
    nx_counts = cluster_counts(LenX)
    ny_counts = cluster_counts(LenY)
    
    Z = np.zeros([N, d])
    # Generate HDGM-D
    start = 0
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i + 283)
        count = nx_counts[i]
        Z[start:start + count, :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, count)
        start += count
    start = LenX
    for i in range(Num_clusters):
        np.random.seed(seed=rs + i)
        sigma_mx_2 = [np.identity(d), np.identity(d)]
        sigma_mx_2[0][0, 1] = 0.5
        sigma_mx_2[0][1, 0] = 0.5
        sigma_mx_2[1][0, 1] = -0.5
        sigma_mx_2[1][1, 0] = -0.5
        count = ny_counts[i]
        Z[start:start + count, :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], count)
        start += count
    np.random.shuffle(Z)
    
    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def sample_HIGGS(N, rs, per, scale):
    """#Feat. 4  #Class 2  #Inst. [5170877,5829123]"""
    np.random.seed(seed=rs)

    data = pickle.load(open('HIGGS_TST.pckl', 'rb'))

    dataX = data[0]
    dataY = data[1]
    del data

    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = dataX[ind1, :4]
    Y = dataY[ind2, :4]
    
    LenX = int(N * per)
    LenY = N-LenX
    # np.random.shuffle(Z)
    Ind_X = np.random.choice(N1_T, LenX, replace=False)
    Ind_Y = np.random.choice(N2_T, LenY, replace=False)
    Z = np.concatenate((dataX[Ind_X,:4],dataY[Ind_Y,:4]), axis=0)
    np.random.shuffle(Z)

    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def sample_MNIST(N, rs, per, scale):
    """#Feat. 28*28  #Class 2  #Inst. [10000,10000]"""
    np.random.seed(seed=rs)

    # True_MNIST
    img_size = 32
    dataloader_FULL_te = torch.utils.data.DataLoader(
        datasets.MNIST(
            "mnist",
            train=False,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(img_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5], [0.5]),
                 ]
            ),
        ),
        batch_size=10000,
        shuffle=True,
    )
    for i, (imgs, Labels) in enumerate(dataloader_FULL_te):
        dataX = np.array(imgs.view(len(imgs), -1))

    # Fake_MNIST
    Fake_MNIST = pickle.load(open('Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
    dataY = torch.from_numpy(Fake_MNIST[0][:])
    dataY = np.array(dataY.view(len(dataY), -1))

    N1_T = dataX.shape[0]
    N2_T = dataY.shape[0]
    ind1 = np.random.choice(N1_T, N, replace=False)
    ind2 = np.random.choice(N2_T, N, replace=False)
    X = dataX[ind1, :]
    Y = dataY[ind2, :]
    
    # """transform to tensor"""
    # X = torch.from_numpy(X)
    # X = X.resize(len(X),1,img_size,img_size)
    LenX = int(N * per)
    LenY = N-LenX
    # np.random.shuffle(Z)
    Ind_X = np.random.choice(N1_T, LenX, replace=False)
    Ind_Y = np.random.choice(N2_T, LenY, replace=False)
    Z = np.concatenate((dataX[Ind_X,:],dataY[Ind_Y,:]), axis=0)
    np.random.shuffle(Z)

    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def sample_CIFAR10(N, rs, per, scale):
    """#Feat. 64*64  #Class 2  #Inst. [10000,2021]"""
    np.random.seed(seed=rs)

    img_size = 32
    dataset_test = datasets.CIFAR10(root='cifar10', download=False, train=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        transforms.Grayscale(),
                                    ]))

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=10000,
                                                  shuffle=True)
    # Obtain CIFAR10 images
    for i, (imgs, Labels) in enumerate(dataloader_test):
        data_all = np.array(imgs.view(len(imgs), -1))

    data_new = np.load('cifar10_X_adversarial.npy')
    data_T = data_new.reshape((-1, 3, img_size, img_size))
    ind_M = np.random.choice(len(data_T), len(data_T), replace=False)
    data_T = data_T[ind_M]
    TT = transforms.Compose([transforms.Resize(img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             transforms.Grayscale(),
                             ])
    trans = transforms.ToPILImage()
    data_trans = torch.zeros([len(data_T), 3, img_size, img_size])
    data_T_tensor = torch.from_numpy(data_T)
    for i in range(len(data_T)):
        d0 = trans(data_T_tensor[i])
        data_trans[i] = TT(d0)
    data_trans = np.array(data_trans.view(len(data_trans), -1))

    Ind = np.random.choice(len(data_all), N, replace=False)
    X = data_all[Ind]
    Ind_v4 = np.random.choice(len(data_trans), N, replace=False)
    Y = data_trans[Ind_v4]

    #"""transform to tensor"""
    # X = torch.from_numpy(X)
    # X = X.resize(len(X),3,img_size,img_size)

    LenX = int(N * per)
    LenY = N-LenX
    Ind_X = np.random.choice(len(data_all), LenX, replace=False)
    Ind_Y = np.random.choice(len(data_trans), LenY, replace=False)
    Z = np.concatenate((data_all[Ind_X],data_trans[Ind_Y]), axis=0)
    np.random.shuffle(Z)

    if scale:
        X, Y, Z = MMscaler(X, Y, Z)
    return X, Y, Z

def load_data(name, N, rs, per, scale=True):
    np.random.seed(seed=1102)
    if name == 'BLOB':
        X, Y, Z = sample_BLOB(N, rs, per, scale)
    elif name == 'HDGM':
        X, Y, Z = sample_HDGM(N, rs, per, scale)
    elif name == 'HIGGS':
        X, Y, Z = sample_HIGGS(N, rs, per, scale)
    elif name == 'MNIST':
        X, Y, Z = sample_MNIST(N, rs, per, scale)   
    elif name == 'CIFAR10':
        X, Y, Z = sample_CIFAR10(N, rs, per, scale)
    elif name == 'CIFAR10_ddpm':
        X, Y, Z = sample_cifar10_ddpm(N, rs, per, scale)
    else:
        print('No Dataset: ', name)
    return X, Y, Z

# X,Y,Z = load_data("CIFAR10_ddpm",200,20)
