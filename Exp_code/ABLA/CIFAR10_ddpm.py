
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
import time
start_time = time.time()
from utils import Ours_method

# parameters to generate data
parser.add_argument('--name',           default='CIFAR10_ddpm', help = 'Dataset')
parser.add_argument('--n_exp',          default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test',         default=100,                   help='Number of test runs')
parser.add_argument('--n_res',          default=100,                   help='Number of resampling runs')
parser.add_argument('--alpha',          default=0.05,                  help='Confidence level of test')
parser.add_argument('--device',         default=torch.device("cuda:0"),  help='Device of data')
parser.add_argument('--dtype',          default=torch.float,           help='Dtype of data')
parser.add_argument('--F_kernel',       default="Deep",           help='type of kernel')
parser.add_argument('--F_select',       default="RST",           help='type of kernel')
parser.add_argument('--F_heur',         default="RST",           help='type of heuristic method')
parser.add_argument('--kernel',         default="Deep",           help='type of kernel')
parser.add_argument('--heur',           default="RST",           help='type of heuristic method')

parser.add_argument('--N1',             default=[20,      40,      60,      80,      100,     120,     140,     160],    help = 'Size of each sample')
parser.add_argument('--rs',             default=[118,     153,     511,     386,     330,     248,     291,     113],    help = 'Random seed')
parser.add_argument('--per',            default=[0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3],    help='Percentage mix; 0.5 for type-I error')

parser.add_argument('--lr_Ours_F',      default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_F',      default=[300,     300,     300,     300,     300,     300,     300,     300],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_F',      default=[32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')
parser.add_argument('--C_Ours_F',       default=[10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4],     help='Coefficient in Ours optimization')

parser.add_argument('--lr_Ours_aug',    default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_aug',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of Ours optimization epochs')
parser.add_argument('--C_Ours_aug',     default=[10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4,  10**-4],     help='Coefficient in Ours optimization')
parser.add_argument('--bs_Ours_aug',    default=[32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')

parser.add_argument('--lr_Ours_non',    default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_non',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_non',    default=[32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')

args = parser.parse_args()

Results = np.zeros((len(args.per), 2, args.n_exp))
Final_results = np.zeros((Results.shape[0],Results.shape[1],2))

Results_P = np.zeros(Results.shape)
Final_results_P = np.zeros(Final_results.shape)

for dd in range(Results.shape[0]):
    H_Ours_aug = np.zeros(args.n_test)
    H_Ours_non = np.zeros(args.n_test)
    for kk in range(args.n_exp):                                               
        H_Ours_aug, _ = Ours_method(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.device, args.dtype, args.kernel, args.heur, args.lr_Ours_aug[dd], args.ne_Ours_aug[dd], args.C_Ours_aug[dd], min(args.bs_Ours_aug[dd],args.N1[dd]), args.F_kernel, args.F_select, args.F_heur,args.lr_Ours_F[dd], args.ne_Ours_F[dd], args.C_Ours_F[dd], min(args.bs_Ours_F[dd],args.N1[dd]))
        # print('Ours_aug Done!')
        
        H_Ours_non, _ = Ours_method(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.device, args.dtype, args.kernel, args.heur, args.lr_Ours_non[dd], args.ne_Ours_non[dd], 0, min(args.bs_Ours_non[dd],args.N1[dd]), args.F_kernel, args.F_select, args.F_heur,args.lr_Ours_F[dd], args.ne_Ours_F[dd], 0, min(args.bs_Ours_F[dd],args.N1[dd]))
        # print('Ours_non Done!')
        
        Results[dd, 0, kk] = H_Ours_aug.sum() / args.n_test
        Results[dd, 1, kk] = H_Ours_non.sum() / args.n_test
        np.savetxt('../../Results/ABLA/'+args.name+'_tp_'+str(args.N1), Results.reshape(Results.shape[0],-1), fmt='%.3f')

    for i in range(Results.shape[1]):
        Final_results[dd][i][0] = Results[dd][i].sum()/args.n_exp
        Final_results[dd][i][1] = Results[dd][i].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/ABLA/'+args.name+'_T_'+str(args.N1), Final_results.reshape(Final_results.shape[0],-1), fmt='%.3f')
# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)