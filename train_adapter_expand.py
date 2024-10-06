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
from scorer import score_folder

def train(hparams, *args):
    if(hparams.CL == "MULTI"):
        hparams.multi = True
        hparams.continual = False
    else:
        hparams.multi = False
        hparams.continual = True

    from single_CL_learner import Seq2SeqToD

    ## travis
    # resume_folder, resume_task_num = check_resume(hparams, task_num=1)
    resume_folder, resume_task_num = check_resume(hparams, resume_task_num=hparams.resume_task_num)

    # max_bottleneck_size = hparams.bottleneck_size
    max_bottleneck_size = None
    if hparams.split_mask:
        calculate_adapter_num(hparams.num_of_adapter)
        # hparams.bottleneck_size = calculate_bottleneck_size(resume_task_num)

    # hparams.bottleneck_size = 50*13
    # hparams.bottleneck_size = min(50*(resume_task_num+1), max_bottleneck_size)
    # hparams.number_of_adpt = resume_task_num+1

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
            logger.info(f"keys is 1 while RUNNING WITH SEED {hparams.seed}")
        else:
            logger.info(f"RUNNING WITH SEED {hparams.seed}")
        for k,_ in train_loader.items():
            print(k)
        # print()


    if resume_folder:
        adapter_task_id = calculate_task_adapter_id(resume_task_num)[-1]
        print(f"adapter_task_id {adapter_task_id}")
        load_checkpoint(model, resume_folder, hparams, resume_task_num, adapter_task_id=adapter_task_id)
        # load_checkpoint(model, resume_folder, hparams, resume_task_num, backbone=True)

    task_seen_so_far = []
    if(hparams.CL != "MULTI"): model.set_number_of_tasks(len(list(train_loader.keys())))
    if(hparams.CL == "GEM"): model.set_up_gem()

    if hparams.multi:
        best_val_loss = np.inf
        best_epoch_idx = 0
        set_requires_grad(model)
        optimizers = configure_optimizers(model)
        cnt = 0
        start_epoch = 0  # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
        for epoch_idx in range(start_epoch, hparams.n_epochs):
            logger.info("Epoch:{}".format(epoch_idx))
            pbar = tqdm(enumerate(train_loader), total=len(train_loader))

            print_diag = False
            train_loss, train_loss_reg = model.train_epoch(None, len(train_loader), pbar, optimizers, hparams,
                                                           print_diag=print_diag)   # task_num=None
            if train_loss_reg:
                logger.debug('Train Loss Reg:{:.3f} '.format(train_loss_reg))

            if (epoch_idx + 1) % int(1) == 0:  # args['evalp']
                print("STARTING EVALUATION")
                pbar = tqdm(enumerate(val_loader), total=len(val_loader))
                valid_loss = model.eval_epoch(None, pbar, hparams)  # task_num=None
                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch_idx = epoch_idx
                    best_model = deepcopy(model.model.state_dict())
                    logger.info('Val Loss:{:.3f} '.format(valid_loss) + " MODEL SAVED")
                    cnt = 0
                else:
                    logger.info('Val Loss:{:.3f}'.format(valid_loss))
                    cnt += 1
                if cnt >= 5:  # 8
                    print("Ran out of patient, early stop...")
                    break

        # Restore best
        model.model.load_state_dict(best_model)
        print('best model reloaded')
        test_model_seq2seq(hparams,model,model.tokenizer,dev_val_loader,time=f"FINAL")

    elif hparams.continual:
        for task_num, (task_id, task_loader) in enumerate(train_loader.items()):
            logger.info('')
            logger.info(f"TASK {task_num}:{task_id}")
            if task_num < resume_task_num:
                model.first_task = False
                task_seen_so_far.append(task_id)
                continue

            adapter_id, adapter_task_id = None, None
            if hparams.split_mask:
                adapter_id, adapter_task_id = calculate_task_adapter_id(task_num)
                logger.info(f"adapter_id: {adapter_id}, adapter_task_id: {adapter_task_id}")
                logger.info(f"bottleneck size: {hparams.bottleneck_size_list[adapter_id]}")

            if task_num > resume_task_num:
                # adapter_task_id = calculate_task_adapter_id(task_num)[-1] if hparams.split_mask else None
                # hparams.bottleneck_size = calculate_bottleneck_size(task_num)
                # hparams.bottleneck_size = min(50 * (task_num + 1), max_bottleneck_size)
                # hparams.number_of_adpt = task_num + 1
                set_seeds(hparams.seed)
                model.init_model(hparams)

                if hparams.CL == "ADAPTER":
                    model.model.adapter_blocks.load_state_dict(best_model)
                    # load_checkpoint(model, save_folder, hparams, task_num, backbone=True)
                    # _, adapter_task_id = calculate_task_adapter_id(task_num)
                    # load_checkpoint(model, save_folder, hparams, task_num, backbone=False, adapter_task_id=adapter_task_id) # max_bottleneck_size=max_bottleneck_size
                else:
                    model.model.load_state_dict(best_model)
                # load_checkpoint(model, save_folder, hparams, task_num)

            if hparams.split_mask:
                # adapter_id, adapter_task_id = calculate_task_adapter_id(task_num)
                # model.model.reset_mask(hparams, task_num)
                if adapter_id > 0 and adapter_task_id == 0:
                    model.init_mask_mem()

            if hparams.todcl_mask or hparams.expand_mask:
                hparams.cur_bottleneck_size = calculate_bottleneck_size(task_num)
                calculate_mask_expand(model, task_num)
                model.model.reset_mask(hparams, task_num)

            set_requires_grad(model)
            optimizers = configure_optimizers(model)

            save_folder = f'{hparams.saving_dir}/{task_num}_{task_id}'
            model.task_list_seen.append(task_id)

            if(hparams.CL == "REPLAY"):
                print(f"Memory Size {len(model.reply_memory)}")
                task_loader = make_loader(hparams,train_datasets[task_id]+model.reply_memory,model.tokenizer)
            if (hparams.CL == "LIMIT-REPLAY"):
                print(f"Memory Size {len(model.reply_memory)}")
                task_loader = make_loader(hparams, train_datasets[task_id] + model.reply_memory, model.tokenizer)
            if(hparams.CL == "LAMOL"):
                # if(current_task_to_load == None or task_num >= current_task_to_load):
                if task_num > 0:
                    number_of_sample = hparams.percentage_LAM0L
                    aug_current_task = get_current_task_data(hparams,train_datasets[task_id],task_id,number_of_sample)
                    print(f"Current {task_id} AUG: {len(aug_current_task)}")
                    aug_data_prev_task = []
                    for task_id_so_far in task_seen_so_far:
                        ## sample data by the LM, priming with [task_id] e.g., [hotel]
                        temp = generate_sample_prev_task(hparams,model.model,model.tokenizer,train_datasets,task_id_so_far,number_of_sample,time=f"{task_num}_{task_id}")
                        print(f"Current {task_id_so_far} AUG: {len(temp)}")
                        aug_data_prev_task += temp
                    ## this task_loader include data generated by the same model
                    task_loader = make_loader(hparams,train_datasets[task_id]+aug_current_task+aug_data_prev_task,model.tokenizer)

            ## CORE
            # start = time.time()
            ### travis
            if hparams.CL == "ADAPTER":
                best_model = {k: v.cpu() for k, v in model.model.adapter_blocks.state_dict().items()}
                # init_state = deepcopy({k: v.cpu() for k, v in model.model.adapter_blocks.state_dict().items()})
            else:
                best_model = {k: v.cpu() for k, v in model.model.state_dict().items()}
                # best_model = deepcopy(model.model.state_dict())

            if hparams.val_retrain:
                retrain_val_data = [val_datasets[val_task_id] for val_task_id in model.task_list_seen]
                retrain_val_data = sum(retrain_val_data, [])
                val_task_loader = make_val_loader(hparams, retrain_val_data, model.tokenizer)
            else:
                val_task_loader = val_loader[task_id]

            best_val_loss = np.inf
            best_epoch_idx = 0
            cnt = 0
            start_epoch = 0 # if we are going to save checkpoint in other folder, then we recalculate the starting epoch
            # if hparams.expand_mask or hparams.todcl_mask:
            #     model.model.reset_mask(hparams, task_num)
            for epoch_idx in range(start_epoch, hparams.n_epochs):
                logger.info("Epoch:{}".format(epoch_idx))
                # for step, batch in enumerate(iter_bar):
                # pbar = tqdm(task_loader, desc='Train Iter (loss=X.XXX)')    # iter_bar
                pbar = tqdm(enumerate(task_loader), total=len(task_loader))

                # if task_num == 1 and epoch_idx == 0:
                #     print_diag = True
                # else:
                #     print_diag = False
                print_diag = False
                train_loss, train_loss_reg = model.train_epoch(task_num, len(task_loader), pbar, optimizers, hparams, print_diag=print_diag)
                # print('Train Loss:{:.3f} '.format(train_loss))
                if train_loss_reg:
                    logger.debug('Train Loss Reg:{:.3f} '.format(train_loss_reg))

                if (epoch_idx + 1) % int(1) == 0:   # args['evalp']
                    print("STARTING EVALUATION")
                    pbar = tqdm(enumerate(val_task_loader), total=len(val_task_loader))
                    valid_loss = model.eval_epoch(task_num, pbar, hparams)
                    if valid_loss < best_val_loss:
                        best_val_loss = valid_loss
                        best_epoch_idx = epoch_idx

                        if hparams.CL == "ADAPTER":
                            best_model = {k: v.cpu() for k, v in model.model.adapter_blocks.state_dict().items()}
                            # save_model(model, save_folder, best_epoch_idx, hparams, save_type='backbone')
                        else:
                            best_model = deepcopy(model.model.state_dict())
                            # best_model = {k: v.cpu() for k, v in model.model.state_dict().items()}

                        logger.info('Val Loss:{:.3f} '.format(valid_loss) + " MODEL SAVED")
                        cnt = 0
                    else:
                        logger.info('Val Loss:{:.3f}'.format(valid_loss))
                        cnt += 1
                    if cnt >= 5:  # 8
                        print("Ran out of patient, early stop...")
                        break

            # Restore best
            if hparams.CL == "ADAPTER":
                model.model.adapter_blocks.load_state_dict(best_model)
                # load_checkpoint(model, save_folder, hparams, task_num, backbone=True, before_retrain=True)
            else:
                model.model.load_state_dict(best_model)
            print('best model reloaded')

            # end = time.time()
            # print ("Time elapsed: %.2fs" % (end - start))

            if hparams.mask:
                mask = calculate_mask(model, task_num)
                # save_model(model, save_folder, best_epoch_idx, hparams, save_type='mask')

            # save to folder
            # save_model(model, save_folder, best_epoch_idx, hparams)
            if hparams.CL == "ADAPTER" or hparams.CL == "VANILLA":
                save_model(model, save_folder, best_epoch_idx, hparams, save_type='backbone')
                if hparams.mask:
                    save_model(model, save_folder, best_epoch_idx, hparams, save_type='mask')

            # test_every_step
            if (hparams.test_every_step):   # and task_num>0):
                if (hparams.CL == "ADAPTER"):
                    if hparams.mask_infer:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}")
                        if task_num > 0 and not hparams.no_TIL:
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                       time=f"{task_num}_{task_id}", TIL=True)
                    elif hparams.mask:    # test with current-task mask
                        if hparams.mask_CIL:    #  and task_num > 0
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                       time=f"{task_num}_{task_id}", masks=model.mask_pre)
                        else:
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                       time=f"{task_num}_{task_id}", masks=mask)
                    elif hparams.single:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}")
                        # test_model_seq2seq_ADAPTER(hparams,model,model.tokenizer,dev_val_loader,test_datasets,time=f"{task_num}_{task_id}", single_task=True)
                    else:
                        test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                   time=f"{task_num}_{task_id}")
                else:
                    test_model_seq2seq(hparams,model,model.tokenizer,dev_val_loader,time=f"{task_num}_{task_id}")

            ## END CORE
            model.first_task = False

            ## retrain and test
            if (task_num > 0) and len(model.episodic_mem.keys()) > 0:
                ### travis: retrain
                if hparams.retrain or hparams.meta:
                    print("STARTING RETRAINING")
                    set_seeds(hparams.seed)

                    # model.init_model(hparams)
                    # model.model.adapter_blocks.load_state_dict(best_model)
                    # load_checkpoint(model, save_folder, hparams, resume_task_num, backbone=True)
                    set_requires_grad(model, retrain=True)

                    optimizers = configure_optimizers(model, retrain=True)
                    # optimizers.zero_grad()

                    retrain_data = []
                    for mem_per_task in model.episodic_mem.values():
                        retrain_data += mem_per_task
                    print(f"Retrain Memory Size {len(retrain_data)}")   #  + Train Data Size {len(train_datasets[task_id])}
                    # retrain_data += train_datasets[task_id]
                    retrain_task_loader = make_loader(hparams, retrain_data, model.tokenizer)

                    cnt = 0
                    # best_model = {k: v.cpu() for k, v in model.model.state_dict().items()} # deepcopy(model.model.state_dict())
                    best_model = {k: v.cpu() for k, v in model.model.adapter_blocks.state_dict().items()}
                    best_retrain_val_loss = best_val_loss   # np.inf
                    retrain_val_task_loader = val_task_loader
                    # if hparams.val_retrain:
                    #     retrain_val_data = [val_datasets[val_task_id] for val_task_id in model.task_list_seen]
                    #     retrain_val_data = sum(retrain_val_data, [])
                    #     retrain_val_task_loader = make_val_loader(hparams, retrain_val_data, model.tokenizer)
                    # else:
                    #     retrain_val_task_loader = val_loader[task_id]

                    cur_task_loader = iter(task_loader)
                    for epoch_idx in range(start_epoch, hparams.retrain_epochs):
                        print("Epoch:{}".format(epoch_idx))
                        pbar = tqdm(enumerate(retrain_task_loader), total=len(retrain_task_loader))

                        if hparams.meta:
                            cur_task_loader = model.meta_train_epoch(task_num, len(retrain_task_loader), pbar, optimizers, hparams, cur_task_loader, task_loader=task_loader)
                        elif hparams.retrain:
                            model.retrain_epoch(task_num, len(retrain_task_loader), pbar, optimizers, hparams)
                        else:
                            raise exit()

                        if (epoch_idx + 1) % int(1) == 0:   # args['evalp']
                            print("STARTING EVALUATION")
                            pbar = tqdm(enumerate(retrain_val_task_loader), total=len(retrain_val_task_loader))
                            valid_loss = model.eval_retrain(task_num, pbar, hparams)
                            if valid_loss < best_retrain_val_loss:
                                # print('Val Loss:{:.3f} '.format(valid_loss), end='')
                                best_retrain_val_loss = valid_loss
                                # best_epoch_idx=epoch_idx
                                # best_model = deepcopy(model.model.state_dict())
                                # best_model = {k: v.cpu() for k, v in model.model.state_dict().items()}
                                best_model = {k: v.cpu() for k, v in model.model.adapter_blocks.state_dict().items()}
                                save_model(model, save_folder, epoch_idx, hparams, save_type='backbone', retrain=True)

                                logger.info('Val Loss:{:.3f} '.format(valid_loss) + " MODEL SAVED")
                                cnt = 0
                            else:
                                logger.info('Val Loss:{:.3f}'.format(valid_loss))
                                cnt += 1
                            if cnt >= 2:  # 5
                                print("Ran out of patient, early stop...")
                                break
                        else:   # 训练定量的epoch，不val
                            cnt += 1
                            # if adapter_id==2 and adapter_task_id==3:
                            #     break
                            if cnt >= hparams.retrain_epochs:
                                break

                    # if hparams.val_retrain:
                    #     model.model.adapter_blocks.load_state_dict(best_model)
                    # else:
                    # model.model.load_state_dict(best_model)
                    model.model.adapter_blocks.load_state_dict(best_model)
                    print('best retrain model reloaded')

                    # save_model(model, save_folder, best_epoch_idx, hparams, retrain=True)

                ### travis: test with all masks
                if (hparams.test_every_step):
                    if hparams.mask:
                        if hparams.retrain:
                            test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                       time=f"{task_num}_{task_id}", retrain=True)
                        # else:
                        #     test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                        #                                time=f"{task_num}_{task_id}", use_all_masks=True,
                        #                                masks=model.mask_pre)
                            if hparams.mask_infer and not hparams.no_TIL:
                                test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                                                           time=f"{task_num}_{task_id}", TIL=True, retrain=True)

                    # else:   # ToDCL TIL
                    #     test_model_seq2seq_ADAPTER(hparams, model, model.tokenizer, dev_val_loader, test_datasets,
                    #                                time=f"{task_num}_{task_id}", TIL=True)

            ## save some training data into the episodic mem
            if hparams.CL == "AGEM":
                # for idx_b, b in enumerate(task_loader):
                #     model.episodic_mem["all"].append(b)
                #     if idx_b == hparams.episodic_mem_size: break
                model.sampling(train_datasets, task_id)
                save_model(model, save_folder, best_epoch_idx, hparams, save_type='memory')
                print('agem')
                agem_data = []
                for mem_per_task in model.episodic_mem.values():
                    agem_data += mem_per_task
                print(f"AGEM Memory Size {len(agem_data)}")
                model.agem_mem_iter = iter(make_loader(hparams, agem_data, model.tokenizer))

            elif hparams.CL == "REPLAY":
                # in percentage
                set_seeds(hparams.seed)
                model.reply_memory += sample(train_datasets[task_id], min(len(train_datasets[task_id]),
                                                                          hparams.episodic_mem_size))  # sample(train_datasets[task_id],min(len(train_datasets[task_id]),int(hparams.episodic_mem_size*len(train_datasets[task_id])))
                save_model(model, save_folder, best_epoch_idx, hparams, save_type='memory')
            elif hparams.CL == "LIMIT-REPLAY":
                set_seeds(hparams.seed)
                size_per_task = hparams.episodic_mem_size // len(model.task_list_seen)
                if model.reply_memory:
                    model.reply_memory = sample(model.reply_memory, min(len(model.reply_memory),
                                                                        hparams.episodic_mem_size - size_per_task))
                    print(f"Old Memory Size {len(model.reply_memory)}")
                model.reply_memory += sample(train_datasets[task_id], min(len(train_datasets[task_id]), size_per_task))
                save_model(model, save_folder, best_epoch_idx, hparams, save_type='memory')
            elif hparams.CL == "EWC":
                # set_seeds(hparams.seed)
                model.sampling(train_datasets, task_id)
                save_model(model, save_folder, best_epoch_idx, hparams, save_type='memory')
                # for idx_b, b in enumerate(task_loader):
                #     model.episodic_mem[task_id].append(b)
                #     if (idx_b+1) == hparams.episodic_mem_size: break
                # print(f"Episodic Memory Size {len(model.episodic_mem[task_id])}")
            else:  ## save example per task
                if hparams.retrain and (task_num < len(train_loader)-1):
                    set_seeds(hparams.seed)
                    model.sampling(train_datasets, task_id, task_num)
                    save_model(model, save_folder, best_epoch_idx, hparams, save_type='memory')

            ##### Compute Fisher info Matrix for EWC
            if hparams.CL == "EWC" or hparams.CL =="L2":
                # model.model.cpu()
                for n, p in model.model.named_parameters():
                    model.optpar[n] = torch.Tensor(p.cpu().data)
                    model.fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()

                if hparams.CL == "EWC":
                    print('optpar and fisher')
                    ewc_data = []
                    for mem_per_task in model.episodic_mem.values():
                        ewc_data += mem_per_task
                    print(f"EWC Memory Size {len(ewc_data)}")
                    ewc_task_loader = make_loader(hparams, ewc_data, model.tokenizer)
                    for _, batch in enumerate(ewc_task_loader):
                        model.model.zero_grad()
                        if USE_CUDA:
                            batch["encoder_input"] = batch["encoder_input"].cuda()
                            batch["decoder_output"] = batch["decoder_output"].cuda()
                        (loss), *_ = model.model(input_ids=batch["encoder_input"],
                                                attention_mask=batch["attention_mask"],
                                                labels=batch["decoder_output"])
                        loss.backward()
                        for n, p in model.model.named_parameters():
                            if p.grad is not None:
                                model.fisher[n].data += p.grad.cpu().data ** 2
                                # model.fisher[n].data += p.grad.data ** 2

                    for name_f,_ in model.fisher.items():
                        model.fisher[name_f] /= len(model.episodic_mem[task_id]) #*hparams.train_batch_size
                    model.model.zero_grad()
            task_seen_so_far.append(task_id)

if __name__ == '__main__':
    train(hparams)
    print(hparams)
    score_folder(hparams)
