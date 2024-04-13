import logging
from ppnp.pytorch.training import train_model_AdaGCN
from ppnp.pytorch.earlystopping import stopping_args
from ppnp.data.io import load_dataset, load_hyper, load_idx
import pandas as pd
import argparse
import torch
from models import HyperAdaGCN #, HyperGCN, HyperGNN
from utils import calc_uncertainty
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
# %% 通用参数
parser.add_argument('--dataset', type=str, default='citeseer',
                    help='ignored when hyperdata used(cora_ml, citeseer, pubmed, ms_academic)')
parser.add_argument('--hyperdata', type=int, default=1, help='0, 1, weather hyperdata is used')
parser.add_argument('--hyper_type', type=str, default='coauthorship', help='coauthorship, cocitation')
parser.add_argument('--hyper_name', type=str, default='cora', help='cora, dblp, citeseer, pubmed')
parser.add_argument('--model', type=str, default='HyperAdaGCN')
parser.add_argument('--ratio', type = float, default = '1', help ='1.0,0.5.0.25')
parser.add_argument('--trainsize', type=int, default=20, help='each class./cora 20/ citeseer 30/ dblp 290/ pubmed 53')
parser.add_argument('--test', type=int, default=1, help='test or not')
parser.add_argument('--niter', type=int, default=1, help='iteration per seed')
parser.add_argument('--nseed', type=int, default=20, help='number of seeds')
parser.add_argument('--train_compare', type=int, default=1, help='0 or 1 weather to train the compare model')
# %% HyperAdaGCN参数
parser.add_argument('--layers', type=int, default=10, help='Number of layers.')
parser.add_argument('--hid_AdaGCN', type=int, default=2500, help='Number of hidden units. default:2500')
parser.add_argument('--lr_AdaGCN', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer(all layers).')
parser.add_argument('--reg', type=float, default=5e-3, help='Weight decay on the 1st layer.')
parser.add_argument('--dropoutadj_AdaGCN', type=float, default=0, help='mixed dropout for adj in AdaGCN.')
parser.add_argument('--dropoutadj_GCN', type=float, default=0, help='mixed dropout for adj in GCN.')
parser.add_argument('--dropout', type=float, default=0, help='ordinary dropout for GCN')
parser.add_argument('--max', type=int, default=500, help='max epoch in early stopping')
parser.add_argument('--patience', type=int, default=500, help='patience in early stopping')
parser.add_argument('--early', type=int, default=1, help='whether early stopping is used')
parser.add_argument('--weighted_voting', type=int, default=1)
# %% (0) initialization
args = parser.parse_args()
type_name = args.hyper_type + args.hyper_name

if type_name == 'coauthorshipcora':
    base_num = 20
elif type_name == 'cocitationcora':
    base_num = 20
elif type_name == 'cocitationciteseer':
    base_num = 24
elif type_name == 'cocitationpubmed':
    base_num = 53
args.trainsize = int(base_num*args.ratio)
print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


stopping_args['max_epochs'] = args.max
stopping_args['patience'] = args.patience
EARLY = True if args.early == 1 else False
logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO)

# %% (1) load data
if args.hyperdata == True:
    graph = load_hyper(args.hyper_name, args.hyper_type)

else:
    graph_name = args.dataset
    graph = load_dataset(graph_name)
    graph.standardize(select_lcc=True)

graph_name = args.dataset
# %% (2) define models: HyperAdaGCN, HyperGCN, f: H —> Z
ALPHA = 0.2 if graph_name == 'ms_academic' else 0.1
if args.model == 'HyperAdaGCN':
    model_class0 = HyperAdaGCN
elif args.model == 'HyperGCN':
    model_class0 = HyperGCN
elif args.model == 'HyperGNN':
    model_class0 = HyperGNN

# %% (3) train parameters
NKNOW = 5000 if args.hyper_name == 'dblp' else 1500
idx_split_args = {'ntrain_per_class': args.trainsize, 'nstopping': 500, 'nknown': NKNOW, 'seed': 1}  # seed: 2413340114
reg_lambda = args.reg
learning_rate = 0.01

model_args = {
    'hiddenunits': [64],
    'drop_prob': args.dropout,
    'propagation': None,  # propagation involves sparse Tensor\
    'lr': learning_rate if args.model == 'GCN' else args.lr_AdaGCN,
    'hid_AdaGCN': args.hid_AdaGCN,
    'layers': args.layers,
    'dropoutadj_GCN': args.dropoutadj_GCN,
    'dropoutadj_AdaGCN': args.dropoutadj_AdaGCN,
    'weight_decay': args.weight_decay,
    'weighted_voting': args.weighted_voting,
}

# %% (4)set seeds
test = True if args.test == 1 else False

test_seeds = [
    2266730403, 2266730442, 2985733717, 2266730404, 1901557222,
    2985733717, 2266730407, 635625077, 3538425002, 2266730409,
    2266730405, 3940842554, 3594628340, 2266730408, 3305901371,
    3644534211, 2297033685, 2266730406, 2590091101, 2266730407]

val_seeds = [
    2413340114, 3258769933, 1789234713, 2222151463, 2813247115,
    1920426428, 4272044734, 2092442742, 841404887, 2188879532,
    646784207, 1633698412, 2256863076, 374355442, 289680769,
    4281139389, 4263036964, 900418539, 119332950, 1628837138]

if test:
    seeds = test_seeds[:args.nseed]
else:
    seeds = val_seeds[:args.nseed]

# %% (5) train
niter_per_seed = args.niter  # random splitting for each seed, default 5
save_result = False
print_interval = 100
results = []
used_seeds = []
niter_tot = niter_per_seed * len(seeds)  # 5 * 20
i_tot = 0
for seed in seeds:
    idx_split_args['seed'] = seed
    idx_num = 1
    # split depends on seed
    for _ in range(niter_per_seed):
        split_idx = load_idx(args.hyper_name, args.hyper_type, str(idx_num))

        i_tot += 1
        logging_string = f"Iteration {i_tot} of {niter_tot}"
        logging.log(22, logging_string + "\n                     " + '-' * len(logging_string))
        # train model
        if args.model == 'HyperAdaGCN':
            result = train_model_AdaGCN(graph_name, args.model, model_class0, graph, model_args, reg_lambda,
                                        idx_split_args, stopping_args, test, device, None, print_interval, EARLY, split_idx)

        # return results
        results.append({})
        results[-1]['stopping_accuracy'] = result['early_stopping']['accuracy']
        results[-1]['stopping_f1_score'] = result['early_stopping']['f1_score']
        results[-1]['valtest_accuracy'] = result['valtest']['accuracy']
        results[-1]['valtest_f1_score'] = result['valtest']['f1_score']
        results[-1]['runtime'] = result['runtime']
        results[-1]['runtime_perepoch'] = result['runtime_perepoch']
        results[-1]['split_seed'] = seed
    idx_num = idx_num +1

# %%(6) evaluation
result_df = pd.DataFrame(results)

result_df.head()
stopping_acc = calc_uncertainty(result_df['stopping_accuracy'])
stopping_f1 = calc_uncertainty(result_df['stopping_f1_score'])
valtest_acc = calc_uncertainty(result_df['valtest_accuracy'])
valtest_f1 = calc_uncertainty(result_df['valtest_f1_score'])
runtime = calc_uncertainty(result_df['runtime'])
runtime_perepoch = calc_uncertainty(result_df['runtime_perepoch'])
import numpy as np
valtest_acc['uncertainty'] = np.std(result_df['valtest_accuracy'])
print("{}\n"
      "Early stopping: Accuracy: {:.2f} ± {:.2f}%, "
      "F1 score: {:.4f} ± {:.4f}\n"
      "{}: Accuracy: {:.2f} ± {:.2f}%, "
      "F1 score: {:.4f} ± {:.4f}\n"
    .format(
    args.model,
    stopping_acc['mean'] * 100,
    stopping_acc['uncertainty'] * 100,
    stopping_f1['mean'],
    stopping_f1['uncertainty'],
    'Test' if test else 'Validation',
    valtest_acc['mean'] * 100,
    valtest_acc['uncertainty'] * 100,
    valtest_f1['mean'],
    valtest_f1['uncertainty'],

))

for i in result_df['valtest_accuracy']:
    print('{:.6f}'.format(i), end=',')
print()
