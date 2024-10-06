import torch
import json
import os
import os.path
import math
import glob
import re

import time
from random import sample
import pytorch_lightning as pl
import random
from pytorch_lightning import Trainer, seed_everything
from utils.dataloader import get_data_loaders, get_current_task_data, make_loader, make_val_loader

from test import test_model_seq2seq, generate_sample_prev_task, test_model_seq2seq_ADAPTER
from collections import defaultdict

from utils.config import *
from tqdm import tqdm
import torch.nn as nn
import shutil
from utils import check_resume, load_checkpoint, save_model
from copy import deepcopy
from utils.utils_CL import set_requires_grad, configure_optimizers, calculate_mask, calculate_mask_expand, calculate_bottleneck_size, calculate_adapter_num, calculate_task_adapter_id
from scorer import score_folder, score_fwt

def train(hparams, *args):
    if(hparams.CL == "MULTI"):
        hparams.multi = True
        hparams.continual = False
    else:
        hparams.multi = False
        hparams.continual = True

    from single_CL_learner import Seq2SeqToD

    ## travis
    hparams.resume_task_num = 1
    resume_folder, resume_task_num = check_resume(hparams, resume_task_num=hparams.resume_task_num)

    # max_bottleneck_size = hparams.bottleneck_size
    max_bottleneck_size = None
    if hparams.split_mask:
        calculate_adapter_num(hparams.num_of_adapter)

    # torch.set_num_threads(1)
    set_seeds(hparams.seed)

    model = Seq2SeqToD(hparams)
    train_loader, val_loader, dev_val_loader, (train_datasets, val_datasets, test_datasets), domains_selected = get_data_loaders(hparams, model.tokenizer)

    ## make the permutation
    if(hparams.continual):
        # seed_everything(hparams.seed)
        if hparams.multi_domain:
            assert domains_selected
            # keys = domains_selected[::-1]
            keys = domains_selected if hparams.test_every_step else domains_selected[::-1]
        elif hparams.fix_dataset:
            keys = ['[\'sgd_weather\']', '[\'sgd_trains\']', '[\'MWOZ_attraction\']']
        # elif hparams.keys1:
            # keys = ['[\'MWOZ_restaurant\']', '[\'MWOZ_hotel\']', '[\'MWOZ_attraction\']', '[\'MWOZ_taxi\']', '[\'MWOZ_train\']']
        else:
            seed = 1 if hparams.keys1 else hparams.seed
            # random.seed(hparams.seed)
            random.seed(seed)
            keys = list(train_loader.keys())
            random.shuffle(keys)

        train_loader = {key: train_loader[key] for key in keys}
        if hparams.keys1:
            print(f"keys is 1 while RUNNING WITH SEED {hparams.seed}")
        else:
            print(f"RUNNING WITH SEED {hparams.seed}")
        for k,_ in train_loader.items():
            print(k)
        # print()

    if resume_folder:
        if hparams.split_mask:
            adapter_task_id = calculate_task_adapter_id(resume_task_num)[-1]
            print(f"adapter_task_id {adapter_task_id}")
            load_checkpoint(model, resume_folder, hparams, resume_task_num, adapter_task_id=adapter_task_id)
        else:
            load_checkpoint(model, resume_folder, hparams, resume_task_num, backbone=True)

    task_seen_so_far = []

    if hparams.continual:
        for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            print()
            print(f"TASK {task_num}:{task_id}")
            if task_num < resume_task_num:
                model.first_task = False
                task_seen_so_far.append(task_id)
                continue

            adapter_id, adapter_task_id = None, None
            if hparams.split_mask:
                adapter_id, adapter_task_id = calculate_task_adapter_id(task_num)
                print(f"adapter_id: {adapter_id}, adapter_task_id: {adapter_task_id}")
                print(f"bottleneck size: {hparams.bottleneck_size_list[adapter_id]}")

            if task_num > resume_task_num:
                set_seeds(hparams.seed)
                model.init_model(hparams)

                if hparams.CL == "ADAPTER":
                    # model.model.adapter_blocks.load_state_dict(best_model)
                    # load_checkpoint(model, save_folder, hparams, task_num, backbone=True)
                    # _, adapter_task_id = calculate_task_adapter_id(task_num)
                    load_checkpoint(model, save_folder, hparams, task_num, backbone=True, adapter_task_id=adapter_task_id) # max_bottleneck_size=max_bottleneck_size
                else:
                    # model.model.load_state_dict(best_model)
                    load_checkpoint(model, save_folder, hparams, task_num, backbone=True)

            if hparams.todcl_mask or hparams.expand_mask:
                hparams.cur_bottleneck_size = calculate_bottleneck_size(task_num)
                calculate_mask_expand(model, task_num)
                model.model.reset_mask(hparams, task_num)

            save_folder = f'{hparams.saving_dir}/{task_num}_{task_id}'
            model.task_list_seen.append(task_id)

            # test_every_step
            if (hparams.test_every_step):   # and task_num>0):
                if (hparams.CL == "ADAPTER"):
                    if hparams.mask_infer:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}", single_task=True)
                        if task_num > 0 and not hparams.no_TIL:
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                       time=f"{task_num}_{task_id}", TIL=True, single_task=True)
                    # elif hparams.mask:    # test with current-task mask
                    #     if hparams.mask_CIL:    #  and task_num > 0
                    #         test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                    #                                    time=f"{task_num}_{task_id}", masks=model.mask_pre)
                    #     else:
                    #         test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                    #                                    time=f"{task_num}_{task_id}", masks=mask)
                    elif hparams.single:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}", single_task=True)
                        # test_model_seq2seq_ADAPTER(hparams,model,model.tokenizer,dev_val_loader,test_datasets,time=f"{task_num}_{task_id}", single_task=True)
                    else:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}", single_task=True)
                else:
                    test_model_seq2seq(hparams,model,model.tokenizer,dev_val_loader,time=f"{task_num}_{task_id}", single_task=True)

            ## END CORE
            model.first_task = False
            task_seen_so_far.append(task_id)

if __name__ == '__main__':
    train(hparams)
    # print(hparams)
    score_fwt(hparams)
