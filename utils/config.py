import os
import argparse
import random
import numpy as np

import torch
import logging
from argparse import ArgumentParser
from utils import set_logger

if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False

def set_seeds(seed):
    # set all possible seeds
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = ArgumentParser()
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('-pd', '--preprocessed_data', action='store_true', help="use preprocessed data")
parser.add_argument('-md', '--multi_domain', type=int, default=None, help="task of multi domain")
parser.add_argument('-bd', "--begin_domain", type=int, default=None, help="first domain idx")
parser.add_argument('-nd', "--num_domain", type=int, default=50, help="Maximum num of multi-domain")
parser.add_argument('-mg', '--multi_gpu', action='store_true')
parser.add_argument('-d', '--distributed', action='store_true')
parser.add_argument('--model_checkpoint', type=str, default="gpt2")
parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=8, help="Batch size for validation")
parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size for validation")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
parser.add_argument("--dataset_list", type=str, default="SGD,TM19,TM20,MWOZ", help="Path for saving")

parser.add_argument('-fix', '--fix_dataset', action='store_true', help="use pre defined dataset")
parser.add_argument('-sd', "--sample_domain", type=int, default=-1, help="number of training samples for each domain")

parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
parser.add_argument("--setting", type=str, default="single", help="Path for saving")
parser.add_argument("--verbose", action='store_true', help="continual baseline")
parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
parser.add_argument("--debug", action='store_true', help="continual baseline")
parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")

parser.add_argument("--bottleneck_size", type=int, default=-1)
parser.add_argument('-todcl', '--todcl_mask', action='store_true', help="expand adapter bottleneck size")
parser.add_argument('-em', '--expand_mask', action='store_true', help="expand adapter bottleneck size")
parser.add_argument('-nm', "--num_of_mask", type=int, default=None, help="number of masks")
# parser.add_argument("--bottleneck_size_per_mask", type=int, default=None, help="bottleneck_size_per_mask")
parser.add_argument("--cur_bottleneck_size", type=int, default=None, help="mask length")

parser.add_argument('-sm', "--split_mask", action='store_true', help="split adapter and masks")
parser.add_argument('-na', "--num_of_adapter", type=int, default=None, help="num_of_adapter")
parser.add_argument("--task_adapter_list", type=list, default=None, help="task_adapter_list")
parser.add_argument("--bottleneck_size_list", type=list, default=None, help="bottleneck_size_list")

parser.add_argument("--number_of_adpt", type=int, default=13, help="number of adapters")
parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
# parser.add_argument("--percentage_LAM0L", type=float, default=0.2, help="LAMOL percentage of augmented data used")
parser.add_argument("--percentage_LAM0L", type=int, default=-1, help="LAMOL percentage of augmented data used")
parser.add_argument("--reg", type=float, default=0.0, help="CL regularization term")    # 0.01
parser.add_argument('-ems', "--episodic_mem_size", type=int, default=-1, help="number of batch/sample put in the episodic memory")
#  options=["E2E","DST","NLG","INTENT"]
parser.add_argument('--task_type', type=str, default="NLG")
#  options=["VANILLA"]
parser.add_argument('--CL', type=str, default="MULTI")
# options=[1,2,3,4,5]
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('-k1', '--keys1', action='store_true', help="seed 1 dataset keys list")
parser.add_argument('-tes', "--test_every_step", action='store_true', help="continual baseline")

parser.add_argument('-s', "--single", action='store_true', help="single adapter")
parser.add_argument('-m', "--mask", action='store_true', help="use HAT mask on adapter")
parser.add_argument("--retrain", action='store_true', help="retrain HAT-masked adapter")
parser.add_argument("--retrain_epochs", type=int, default=1, help="Number of retraining epochs")

parser.add_argument("--meta", action='store_true', help="meta train HAT-masked adapter")
parser.add_argument('-fu', '--fast_update', help='meta step', type=int, required=False, default=1)
parser.add_argument('-mqs', '--meta_query_step', help='meta query step', type=int, required=False, default=1)
parser.add_argument('-de', '--direction', help='num of direction', type=int, required=False, default=2)

# parser.add_argument('-t1', '--task1', action='store_true', help='start training from second task')
parser.add_argument('-t', '--resume_task_num', help='start training from reuse_task', type=int, required=False, default=0)
parser.add_argument('-r', '--reuse_model', action='store_true', help='reuse existing model')
parser.add_argument('-cm', '--cumulative_mask', action='store_true', help='grad update with cumulative mask during retrain')
parser.add_argument('-v', "--val_retrain", action='store_true', help="val on model after retrain")
parser.add_argument('-rlr', "--retrain_lr_factor", type=float, default=0.1, help="Number of retraining epochs")
parser.add_argument("--retrain_gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
parser.add_argument('-u', "--uncertainty", action='store_true', help="select memory by uncertainty")
parser.add_argument('-mc', "--mask_CIL", action='store_true', help="use previous+current HAT mask on adapter")
parser.add_argument('-f', "--freeze_emb", action='store_true', help="do not train wte/wpe embedding")
parser.add_argument('-mi', "--mask_infer", action='store_true', help="infer mask at test")
parser.add_argument('-b', "--balance", action='store_true', help="data balance loss for meta")
parser.add_argument('-l', "--lamb", action='store_true', help="change lamb of each task")
parser.add_argument('-aug', "--augmentation", type=int, default=None, help="num of augmentation times for sample")
parser.add_argument("--saving_dir", type=str, default="", help="Path to the folder with the results")

parser.add_argument("-notil", "--no_TIL", action='store_true', help="do not generate TIL")
parser.add_argument("-fwt", "--fwt", action='store_true', help="run fwt")
parser.add_argument("-bwt", "--bwt", action='store_true', help="calculate score_bwt")

# os.environ['CUDA_VISIBLE_DEVICES'] = '6' # '0,5'

hparams = parser.parse_args()   # hyperparams

if hparams.saving_dir == "":
    appendix = ''
    if hparams.begin_domain:
        appendix += f'bd{hparams.begin_domain}_'
    if(hparams.CL == "ADAPTER"):
        if hparams.augmentation:
            appendix += f'aug{hparams.augmentation}_'
        if hparams.num_of_mask:
            appendix += f'nm{hparams.num_of_mask}_'
        if hparams.num_of_adapter:
            appendix += f'na{hparams.num_of_adapter}_'
        # if hparams.uncertainty:
        #     appendix += 'uncert_'
        if hparams.cumulative_mask:
            appendix += 'cm-ewc_'
        # if hparams.meta:
        #     appendix += 'meta_'
        if hparams.retrain:
            appendix += f'retrain{hparams.retrain_epochs}_'
        # if hparams.mask:
        #     appendix += 'mask_'
        # if hparams.single:
        #     appendix += 'single_'
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}{'_mul' if hparams.multi_domain else ''}/{appendix}{hparams.CL}_EM_{hparams.episodic_mem_size}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    else:
        hparams.saving_dir = f"runs_{hparams.task_type}/{hparams.dataset_list}{'_mul' if hparams.multi_domain else ''}/{hparams.CL}_{appendix}EM_{hparams.episodic_mem_size}_LAMOL_{hparams.percentage_LAM0L}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"

# print(vars(parser.parse_args()))
print(hparams)

if not os.path.exists(hparams.saving_dir):
    os.makedirs(hparams.saving_dir)

if hparams.mode == 'train':
    logger = set_logger(hparams.saving_dir, hparams.resume_task_num)
    print('logger init')

# domains_selected=None