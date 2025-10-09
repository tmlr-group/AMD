import torch
import argparse
parser = argparse.ArgumentParser()
from torchvision import models
import numpy as np
from utils import *
import time
start_time = time.time()
# parameters of experimental setting
parser.add_argument('--n_exp',     default=10,              help='Number of experiment runs')
parser.add_argument('--n_test',    default=100,             help='Number of two-sample test runs')
parser.add_argument('--n_res',     default=100,                   help='Number of resampling runs')
parser.add_argument('--alpha',     default=0.05,            help='Confidence level of two-sample test')
parser.add_argument('--device',    default=torch.device("cuda"),  help='Device of data')
parser.add_argument('--dtype',     default=torch.float,          help='Dtype of data')
parser.add_argument('--F_kernel',  default="Gaussian",           help='type of kernel')
parser.add_argument('--F_select',  default="TST",           help='type of kernel')
parser.add_argument('--F_heur',    default="TST",           help='type of heuristic method')
parser.add_argument('--kernel',    default="Gaussian",           help='type of kernel')
parser.add_argument('--heur',      default="RST",           help='type of heuristic method')

# parameters of experimental setting
parser.add_argument('--N1',        default=[25,      50,      75,      100,     125,     150,     175,     200,     225,     250],    help = 'Size of each sample in optimization')
parser.add_argument('--rs',        default=[183,     183,     183,     184,     186,     606,     183,     186,     185,     197],  help = 'Random seed')

# parameters of MMD
parser.add_argument('--lr_MMD',    default=[0.0005,  0.0005,  0.0005,  0.0005,  0.0005,  0.0005,  0.0005,  0.0005,  0.0005,  0.0005], help='Learning rate of MMD')
parser.add_argument('--ne_MMD',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD',    default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in MMD optimization')

# parameters of Ours
parser.add_argument('--lr_Ours_F',  default=[0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_F',  default=[300,     300,     300,     300,     300,     300,     300,     300,     300,     300],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_F',  default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')
parser.add_argument('--C_Ours_F',   default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')

parser.add_argument('--lr_Ours',    default=[0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1], help='Learning rate of Ours')
parser.add_argument('--ne_Ours',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of Ours optimization epochs')
parser.add_argument('--C_Ours',     default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')
parser.add_argument('--bs_Ours',    default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')

args = parser.parse_args()

resnet50 = models.resnet50(pretrained=True).cuda()
resnet50.eval()

Z = torch.load('imagenet_Fea.pt')
Y = torch.load('imagenetv2_Fea.pt')
X = torch.load('imageneta_Fea.pt')

Results = np.zeros((len(args.N1), 2, args.n_exp))
Final_results = np.zeros((Results.shape[0], Results.shape[1], 2))
Results_P = np.zeros(Results.shape)
Final_results_P = np.zeros(Final_results.shape)

for dd in range(len(args.N1)):
    np.random.seed(seed=args.rs[dd])
    torch.manual_seed(args.rs[dd])
    torch.cuda.manual_seed(args.rs[dd])
    for kk in range(args.n_exp):
        indZ = np.random.choice(len(Z), args.N1[dd], replace=False)
        indY = np.random.choice(len(Y), args.N1[dd], replace=False)
        indX = np.random.choice(len(X), args.N1[dd], replace=False)
    
        H_Ours = np.zeros(args.n_test)
        P_Ours = np.zeros(args.n_test)
        H_Ours, P_Ours  = Ours_method(None, args.N1[dd], args.rs[dd]+kk+2, args.n_test, args.n_res, args.alpha, args.device, args.dtype, X[indX], Y[indY], Z[indZ], X, Y, Z, args.kernel, args.heur, args.lr_Ours[dd], args.ne_Ours[dd], args.C_Ours[dd], args.bs_Ours[dd], args.F_kernel, args.F_select, args.F_heur,args.lr_Ours_F[dd], args.ne_Ours_F[dd], args.C_Ours_F[dd], args.bs_Ours_F[dd])
        
        H_MMD = np.zeros(args.n_test)
        P_MMD = np.zeros(args.n_test)
        sigma0_MMD, M_matrix_MMD = training_MMD(args.kernel, args.N1[dd], X[indX], Y[indY], Z[indZ], args.lr_MMD[dd], args.ne_MMD[dd], args.bs_Ours[dd], args.device, args.dtype)
        H_MMD, P_MMD  = MMD_test(args.kernel, X, Y, Z, args.N1[dd], args.n_test, args.alpha, args.rs[dd]+kk+2, sigma0_MMD, M_matrix_MMD)
        
        Results[dd, 0, kk] = H_Ours.sum() / args.n_test
        Results[dd, 1, kk] = H_MMD.sum() / args.n_test
        Results_P[dd, 0, kk] = P_Ours.sum() / args.n_test
        Results_P[dd, 1, kk] = P_MMD.sum() / args.n_test
        np.savetxt('../../Results/DA/'+'v2_a_tp_'+str(args.N1), Results.reshape(Results.shape[0],-1), fmt='%.3f')
        # np.savetxt('../../Results/DA/'+'v2_a_pv_'+str(args.N1), Results_P.reshape(Results_P.shape[0],-1), fmt='%.10f')
    
    Final_results[dd][0][0] = Results[dd][0].sum()/args.n_exp
    Final_results[dd][0][1] = Results[dd][0].std()/np.sqrt(args.n_exp)
    Final_results[dd][1][0] = Results[dd][1].sum()/args.n_exp
    Final_results[dd][1][1] = Results[dd][1].std()/np.sqrt(args.n_exp)
    
    Final_results_P[dd][0][0] = Results_P[dd][0].sum()/args.n_exp
    Final_results_P[dd][0][1] = Results_P[dd][0].std()/np.sqrt(args.n_exp)
    Final_results_P[dd][1][0] = Results_P[dd][1].sum()/args.n_exp
    Final_results_P[dd][1][1] = Results_P[dd][1].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/DA/'+'v2_a_T_'+str(args.N1), Final_results.reshape(Final_results.shape[0],-1), fmt='%.3f')
    # np.savetxt('../../Results/DA/'+'v2_a_P_'+str(args.N1), Final_results_P.reshape(Final_results_P.shape[0],-1), fmt='%.10f')

# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)