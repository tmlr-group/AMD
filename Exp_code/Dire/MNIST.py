import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from utils import Ours_method
import time
start_time = time.time()

# parameters to generate data
parser.add_argument('--name',           default='MNIST', help = 'Dataset')
parser.add_argument('--n_exp',          default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test',         default=100,                   help='Number of test runs')
parser.add_argument('--device',         default=torch.device("cuda:0"),  help='Device of data')
parser.add_argument('--dtype',          default=torch.float,           help='Dtype of data')
parser.add_argument('--F_kernel',       default="Laplace",           help='type of kernel')
parser.add_argument('--F_select',       default="RST",           help='type of kernel')
parser.add_argument('--F_heur',         default="RST",           help='type of heuristic method')

parser.add_argument('--N1',             default=[50,      100,     150,     200,     250,     300,     350,     400],    help = 'Size of each sample')
parser.add_argument('--rs',             default=[1000,    2000,    3000,    4000,    5000,    6000,    7000,    8000],    help = 'Random seed')
parser.add_argument('--per',            default=[0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3,     0.3],    help='Percentage mix; 0.5 for type-I error')

# parameters of Ours
parser.add_argument('--lr_Ours_F',      default=[0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_F',      default=[300,     300,     300,     300,     300,     300,     300,     300],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_F',      default=[32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')
parser.add_argument('--C_Ours_F',       default=[5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')

args = parser.parse_args()

Results = np.zeros((len(args.N1), args.n_exp))
Final_results = np.zeros((Results.shape[0],2))

for dd in range(Results.shape[0]):
    for kk in range(args.n_exp):
        H_Ours_F = np.zeros(args.n_test)                                                  
        H_Ours_F = Ours_method(args.name, args.F_kernel, args.N1[dd], kk*args.n_test+args.rs[dd], args.n_test, args.per[dd], args.device, args.dtype, args.C_Ours_F[dd], args.lr_Ours_F[dd], args.ne_Ours_F[dd], min(args.bs_Ours_F[dd], args.N1[dd]), args.F_select, args.F_heur)

        Results[dd,kk] = H_Ours_F.sum() / args.n_test
        np.savetxt('../../Results/Dire/'+args.F_select+'_'+args.F_heur+'_'+args.name+'_m_'+str(args.N1), Results, fmt='%.3f')
    
    Final_results[dd][0] = Results[dd].sum() / args.n_exp
    Final_results[dd][1] = Results[dd].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/Dire/'+args.F_select+'_'+args.F_heur+'_'+args.name+str(args.N1), Final_results, fmt='%.3f')
    print("Ours_F: {:.3f}±{:.3f}".format(Final_results[dd][0], Final_results[dd][1]), flush=True)
    
# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)