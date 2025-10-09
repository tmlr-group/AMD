
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

# parameters to generate data
parser.add_argument('--name',           default='CIFAR10_ddpm', help = 'Dataset')
parser.add_argument('--n_exp',          default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test',         default=100,                   help='Number of test runs')
parser.add_argument('--n_res',          default=100,                   help='Number of resampling runs')
parser.add_argument('--alpha',          default=0.05,                  help='Confidence level of test')
parser.add_argument('--device',         default=torch.device("cuda:0"),  help='Device of data')
parser.add_argument('--dtype',          default=torch.float,           help='Dtype of data')
parser.add_argument('--F_kernel',       default="Laplace",           help='type of kernel')
parser.add_argument('--F_select',       default="RST",           help='type of kernel')
parser.add_argument('--F_heur',         default="RST",           help='type of heuristic method')
parser.add_argument('--kernel',         default="Deep",           help='type of kernel')
parser.add_argument('--heur',           default="RST",           help='type of heuristic method')

parser.add_argument('--N1',             default=[50,      50,      50,      50,      50,      50,      50,      50,      50,      50,      50],    help = 'Size of each sample')
parser.add_argument('--rs',             default=[44,      44,      44,      74,      113,     44,      74,      164,     54,      54,      54],    help = 'Random seed')
parser.add_argument('--per',            default=[0.0,     0.1,     0.2,     0.3,     0.4,     0.5,     0.6,     0.7,     0.8,     0.9,     1.0],    help='Percentage mix; 0.5 for type-I error')

# parameters of UME
parser.add_argument('--lr_UME',         default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of UME')
parser.add_argument('--ne_UME',         default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of UME optimization epochs')
parser.add_argument('--bs_UME',         default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in UME optimization')
parser.add_argument('--v_UME',          default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='number of test locations')

# parameters of SCHE_LBI
parser.add_argument('--lr_SCHE_LBI',    default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of SCHE_LBI')
parser.add_argument('--ne_SCHE_LBI',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of SCHE_LBI optimization epochs')
parser.add_argument('--bs_SCHE_LBI',    default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in SCHE_LBI optimization')
parser.add_argument('--scale_coe_SL',   default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5,       5],   help='scale_coe')

# parameters of MMD
parser.add_argument('--lr_MMD',         default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of MMD')
parser.add_argument('--ne_MMD',         default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of MMD optimization epochs')
parser.add_argument('--bs_MMD',         default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in MMD optimization')

# parameters of KLFI
parser.add_argument('--lr_KLFI',        default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of KLFI')
parser.add_argument('--ne_KLFI',        default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of KLFI optimization epochs')
parser.add_argument('--bs_KLFI',        default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in KLFI optimization')
parser.add_argument('--scale_coe_KLFI', default=[1,       1,       1,       1,       1,       1,       1,       1,       1,       1,       1],    help='scale_coe')

# parameters of Ours
parser.add_argument('--lr_Ours_F',      default=[0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1,     0.1], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_F',      default=[300,     300,     300,     300,     300,     300,     300,     300,     300,     300,     300],   help='Number of Ours optimization epochs')
parser.add_argument('--bs_Ours_F',      default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')
parser.add_argument('--C_Ours_F',       default=[5,       5,       5,       5,       5,       5,       5,       5,       5,       5,       5],     help='Coefficient in Ours optimization')

parser.add_argument('--lr_Ours_ori',    default=[0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005], help='Learning rate of Ours')
parser.add_argument('--ne_Ours_ori',    default=[1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000,    1000],   help='Number of Ours optimization epochs')
parser.add_argument('--C_Ours_ori',     default=[5,       5,       0,       5,       5,       5,       0,       5,       0,       5,       5],     help='Coefficient in Ours optimization')
parser.add_argument('--bs_Ours_ori',    default=[32,      32,      32,      32,      32,      32,      32,      32,      32,      32,      32],   help='Batch size in Ours optimization')

args = parser.parse_args()

Results = np.zeros((len(args.per), 8, args.n_exp))
Final_results = np.zeros((Results.shape[0],Results.shape[1],2))

Results_P = np.zeros(Results.shape)
Final_results_P = np.zeros(Final_results.shape)

for dd in range(Results.shape[0]):
    H_UME = np.zeros(args.n_test)
    H_sche = np.zeros(args.n_test)
    H_like = np.zeros(args.n_test)
    H_MMD_deep = np.zeros(args.n_test)
    H_MMD_heur = np.zeros(args.n_test)
    H_KLFI = np.zeros(args.n_test)
    H_Ours_ori = np.zeros(args.n_test)
    H_Ours_ts = np.zeros(args.n_test)
    P_UME = np.zeros(args.n_test)
    P_sche = np.zeros(args.n_test)
    P_like = np.zeros(args.n_test)
    P_MMD_deep = np.zeros(args.n_test)
    P_MMD_heur = np.zeros(args.n_test)
    P_KLFI = np.zeros(args.n_test)
    P_Ours_ori = np.zeros(args.n_test)
    P_Ours_ts = np.zeros(args.n_test)
    
    for kk in range(args.n_exp):                                               
        # # # UME
        from baseline_tests.UME_wild import UME_method
        H_UME, P_UME = UME_method(args.name, args.N1[dd], kk+args.rs[dd], args.v_UME[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.device, args.dtype, args.lr_UME[dd], args.ne_UME[dd], args.bs_UME[dd])
        # print('UME Done!')

        # # # SCHE_LBI
        from baseline_tests.SCHE_LBI import SCHE_LBI_method
        H_sche, P_sche, H_like, P_like = SCHE_LBI_method(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.scale_coe_SL[dd], args.device, args.dtype, args.lr_SCHE_LBI[dd], args.ne_SCHE_LBI[dd], args.bs_SCHE_LBI[dd])
        # print('SCHE_LBI_method Done!')

        # # # MMD_deep
        from baseline_tests.MMD_asym import MMD_method_deep
        H_MMD_deep, P_MMD_deep = MMD_method_deep(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.per[dd], args.alpha, args.device, args.dtype, args.lr_MMD[dd], args.ne_MMD[dd], args.bs_MMD[dd])
        # print('MMD_deep Done!')

        # # # MMD_heur
        from baseline_tests.MMD_asym import MMD_method_heur
        H_MMD_heur, P_MMD_heur = MMD_method_heur(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.per[dd], args.alpha, args.device, args.dtype)
        # print('MMD_heur Done!')

        # # # KLFI
        from baseline_tests.KLFI import KLFI_method
        H_KLFI, P_KLFI = KLFI_method(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.scale_coe_KLFI[dd], args.device, args.dtype, args.lr_KLFI[dd], args.ne_KLFI[dd], args.bs_KLFI[dd])
        # print('KLFI Done!')
        
        # # # Ours
        from utils import Ours_method
        H_Ours_ori, P_Ours_ori, H_Ours_ts, P_Ours_ts = Ours_method(args.name, args.N1[dd], kk+args.rs[dd], args.n_test, args.n_res, args.per[dd], args.alpha, args.device, args.dtype, args.kernel, args.heur, args.lr_Ours_ori[dd], args.ne_Ours_ori[dd], args.C_Ours_ori[dd], args.bs_Ours_ori[dd], args.F_kernel, args.F_select, args.F_heur,args.lr_Ours_F[dd], args.ne_Ours_F[dd], args.C_Ours_F[dd], args.bs_Ours_F[dd])
        # print('Ours Done!')
        
        Results[dd, 0, kk] = H_UME.sum() / args.n_test
        Results[dd, 1, kk] = H_sche.sum() / args.n_test
        Results[dd, 2, kk] = H_like.sum() / args.n_test
        Results[dd, 3, kk] = H_MMD_deep.sum() / args.n_test
        Results[dd, 4, kk] = H_MMD_heur.sum() / args.n_test
        Results[dd, 5, kk] = H_KLFI.sum() / args.n_test
        Results[dd, 6, kk] = H_Ours_ori.sum() / args.n_test
        Results[dd, 7, kk] = H_Ours_ts.sum() / args.n_test
        np.savetxt('../../Results/base/'+args.name+'_tp_'+str(args.N1), Results.reshape(Results.shape[0],-1), fmt='%.3f')
        
        Results_P[dd, 0, kk] = P_UME.sum() / args.n_test
        Results_P[dd, 1, kk] = P_sche.sum() / args.n_test
        Results_P[dd, 2, kk] = P_like.sum() / args.n_test
        Results_P[dd, 3, kk] = P_MMD_deep.sum() / args.n_test
        Results_P[dd, 4, kk] = P_MMD_heur.sum() / args.n_test
        Results_P[dd, 5, kk] = P_KLFI.sum() / args.n_test
        Results_P[dd, 6, kk] = P_Ours_ori.sum() / args.n_test
        Results_P[dd, 7, kk] = P_Ours_ts.sum() / args.n_test
        np.savetxt('../../Results/base/'+args.name+'_pv_'+str(args.N1), Results_P.reshape(Results_P.shape[0],-1), fmt='%.10f')

    for i in range(Results.shape[1]):
        Final_results[dd][i][0] = Results[dd][i].sum()/args.n_exp
        Final_results[dd][i][1] = Results[dd][i].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/base/'+args.name+'_T_'+str(args.N1), Final_results.reshape(Final_results.shape[0],-1), fmt='%.3f')
    
    for i in range(Results_P.shape[1]):
        Final_results_P[dd][i][0] = Results_P[dd][i].sum()/args.n_exp
        Final_results_P[dd][i][1] = Results_P[dd][i].std()/np.sqrt(args.n_exp)
    np.savetxt('../../Results/base/'+args.name+'_P_'+str(args.N1), Final_results_P.reshape(Final_results_P.shape[0],-1), fmt='%.10f')
        

# End timing
end_time = time.time()
# Calculate execution time
execution_time = end_time - start_time
print(f"Program execution time: {execution_time} seconds", flush=True)