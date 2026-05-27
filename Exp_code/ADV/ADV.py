import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from utils import *
from train_model import *
from adv_generator import * 
import time
start_time = time.time()

parser.add_argument('--perturb_steps',  default=20,           help='perturb_steps')
parser.add_argument('--loss_fn',    default="cent",           help='loss_fn')
parser.add_argument('--category',   default="Madry",           help='category')
parser.add_argument('--net',        default="resnet18",       help='choose from resnet18, resnet34')
parser.add_argument('--dataset',    default="cifar10",        help='choose from cifar10, svhn')
parser.add_argument('--model_path', default="./Res18_model/net_150.pth", help='model checkpoint for adversarial data generation')

# parameters of experimental setting
parser.add_argument('--n_exp',     default=10,              help='Number of experiment runs')
parser.add_argument('--n_test',    default=100,             help='Number of two-sample test runs')
parser.add_argument('--n_res',     default=100,                   help='Number of resampling runs')
parser.add_argument('--alpha',     default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--device',    default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',     default=torch.float,          help='Dtype of data')
parser.add_argument('--F_kernel',  default="Laplace",           help='type of kernel')
parser.add_argument('--F_select',  default="RST",           help='type of kernel')
parser.add_argument('--F_heur',    default="RST",           help='type of heuristic method')
parser.add_argument('--kernel',    default="Gaussian",           help='type of kernel')
parser.add_argument('--heur',      default="RST",           help='type of heuristic method')

# parameters of experimental setting
parser.add_argument('--epss',      default=[1,       2,       3,       4,       5,       6,       7,       8,       9,       10],    help = 'Size of each sample in optimization')
parser.add_argument('--N1',        default=[170,     170,     170,     170,     170,     170,     170,     170,     170,     170],    help = 'Size of each sample in optimization')
parser.add_argument('--rs',        default=[183,     183,     183,     183,     183,     183,     183,     183,     183,     221],  help = 'Random seed')

# parameters of MMD
parser.add_argument('--lr_MMD',    default=[0.001,   0.001,   0.001,   0.001,   0.001,   0.001,   0.001,   0.001,   0.001,   0.001], help='Learning rate of MMD')
parser.add_argument('--ne_MMD',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD',    default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in MMD optimization')


# parameters of Ours
parser.add_argument('--lr_Ours_F',  default=[0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_F',  default=[300,     300,     300,     300,     300,     300,     300,     300,     300,     300],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_F',  default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')
parser.add_argument('--C_Ours_F',   default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')

parser.add_argument('--lr_Ours',    default=[1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0,     1.0], help='Learning rate of Ours')
parser.add_argument('--ne_Ours',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of Ours optimization epochs')
parser.add_argument('--C_Ours',     default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')
parser.add_argument('--bs_Ours',    default=[32,      32,      32,     32,       32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')

args = parser.parse_args()

if args.net == "resnet18":
    model = ResNet18_Fea().to(args.device)
    net = "resnet18"
elif args.net == "resnet34":
    model = ResNet34_Fea().to(args.device)
    net = "resnet34"
else:
    raise ValueError("net must be either 'resnet18' or 'resnet34'.")

ckpt = torch.load(args.model_path, map_location=args.device)
model.load_state_dict(ckpt)

np.random.seed(seed=1102)
torch.manual_seed(1102)
torch.cuda.manual_seed(1102)

transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=0)
for Z, target in test_loader:
    Z, target = Z.detach().to(args.device), target.detach().to(args.device)
num_classes = 10
one_hot = torch.zeros(target.size(0), num_classes, device=target.device)
one_hot.scatter_(1, target.unsqueeze(1), 1)
ww = one_hot @ model.fc.weight.detach().to(args.device)
Z_Fea = torch.zeros(ww.shape).to(args.device)
for i in range(20):
    Z_Fea[i*500:(i+1)*500] = model(Z[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
del Z

eps = args.epss[3]
Y = adv_generator(
    args.perturb_steps,
    epsilon=eps/255,
    step_size=eps/255/20,
    loss_fn=args.loss_fn,
    category=args.category,
    net=args.net,
    dataset=args.dataset,
    model_path=args.model_path,
    device=args.device,
).detach().to(args.device)
Y_Fea = torch.zeros(ww.shape).to(args.device)
for i in range(20):
    Y_Fea[i*500:(i+1)*500] = model(Y[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
del Y

Results = np.zeros((len(args.N1), 2, args.n_exp))
Final_results = np.zeros((Results.shape[0], Results.shape[1], 2))
Results_P = np.zeros(Results.shape)
Final_results_P = np.zeros(Final_results.shape)

for dd in range(len(args.N1)):
    eps = args.epss[dd]
    X = adv_generator(
        args.perturb_steps,
        epsilon=eps/255,
        step_size=eps/255/20,
        loss_fn=args.loss_fn,
        category=args.category,
        net=args.net,
        dataset=args.dataset,
        model_path=args.model_path,
        device=args.device,
    ).detach().to(args.device)
    X_Fea = torch.zeros(ww.shape).to(args.device)
    for i in range(20):
        X_Fea[i*500:(i+1)*500] = model(X[i*500:(i+1)*500]).detach().to(args.device).reshape(500,-1) * ww[i*500:(i+1)*500].detach()
    del X

    np.random.seed(seed=args.rs[dd])
    torch.manual_seed(args.rs[dd])
    torch.cuda.manual_seed(args.rs[dd])
    for kk in range(args.n_exp):
        ind = np.random.choice(len(Z_Fea), args.N1[dd], replace=False)
        H_Ours = np.zeros(args.n_test)
        P_Ours = np.zeros(args.n_test)
        H_Ours, P_Ours  = Ours_method(None, args.N1[dd], args.rs[dd]+kk+2, args.n_test, args.n_res, args.alpha, args.device, args.dtype, X_Fea[ind], Y_Fea[ind], Z_Fea[ind], X_Fea, Y_Fea, Z_Fea, args.kernel, args.heur, args.lr_Ours[dd], args.ne_Ours[dd], args.C_Ours[dd], args.bs_Ours[dd], args.F_kernel, args.F_select, args.F_heur,args.lr_Ours_F[dd], args.ne_Ours_F[dd], args.C_Ours_F[dd], args.bs_Ours_F[dd])
        
        H_MMD = np.zeros(args.n_test)
        P_MMD = np.zeros(args.n_test)
        sigma0_MMD, M_matrix_MMD = training_MMD(args.kernel, args.N1[dd], X_Fea[ind], Y_Fea[ind], Z_Fea[ind], args.lr_MMD[dd], args.ne_MMD[dd], args.bs_Ours[dd], args.device, args.dtype)
        H_MMD, P_MMD  = MMD_test(args.kernel, X_Fea, Y_Fea, Z_Fea, args.N1[dd], args.n_test, args.alpha, args.rs[dd]+kk+2, sigma0_MMD, M_matrix_MMD)
        
        Results[dd, 0, kk] = H_Ours.sum() / args.n_test
        Results[dd, 1, kk] = H_MMD.sum() / args.n_test
        Results_P[dd, 0, kk] = P_Ours.sum() / args.n_test
        Results_P[dd, 1, kk] = P_MMD.sum() / args.n_test
        np.savetxt('../../Results/ADV/'+'tp_'+str(args.N1), Results.reshape(Results.shape[0],-1), fmt='%.3f')
        np.savetxt('../../Results/ADV/'+'pv_'+str(args.N1), Results_P.reshape(Results_P.shape[0],-1), fmt='%.10f')
        
    Final_results[dd][0][0] = Results[dd][0].sum()/args.n_exp
    Final_results[dd][0][1] = Results[dd][0].std()/np.sqrt(args.n_exp)
    Final_results[dd][1][0] = Results[dd][1].sum()/args.n_exp
    Final_results[dd][1][1] = Results[dd][1].std()/np.sqrt(args.n_exp)
    
    Final_results_P[dd][0][0] = Results_P[dd][0].sum()/args.n_exp
    Final_results_P[dd][0][1] = Results_P[dd][0].std()/np.sqrt(args.n_exp)
    Final_results_P[dd][1][0] = Results_P[dd][1].sum()/args.n_exp
    Final_results_P[dd][1][1] = Results_P[dd][1].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/ADV/'+'T_'+str(args.N1), Final_results.reshape(Final_results.shape[0],-1), fmt='%.3f')
    np.savetxt('../../Results/ADV/'+'P_'+str(args.N1), Final_results_P.reshape(Final_results_P.shape[0],-1), fmt='%.10f')

# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)
