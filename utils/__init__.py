import sys
import torch
import logging
import os
import glob
import numpy as np
import shutil
# from utils.config import *
import pandas as pd
from copy import deepcopy
# from utils.utils_CL import calculate_task_adapter_id

class Optimizers(object):
    def __init__(self):
        self.optimizers = []
        self.lrs = []

    def add(self, optimizer, lr):
        self.optimizers.append(optimizer)
        self.lrs.append(lr)

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def __getitem__(self, index):
        return self.optimizers[index]

    def __setitem__(self, index, value):
        self.optimizers[index] = value


class Schedulers(object):
    def __init__(self):
        self.schedulers = []

    def add(self, scheduler):
        self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def __getitem__(self, index):
        return self.schedulers[index]

    def __setitem__(self, index, value):
        self.schedulers[index] = value


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        return

    def update(self, val, num):
        self.sum += val.mean() * num
        self.n += num

    @property
    def avg(self):
        return self.sum / self.n


class CsvLogger(object):
    def __init__(self, file_name='logger', resume=False, path='./csvdata/', data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format

    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        # self.log = self.log.append(df, ignore_index=True)
        self.log = pd.concat([self.log, df], ignore_index=True)

    def add_list(self, dict, columns, index=None):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        # df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        df = pd.DataFrame(dict, index=index, columns=columns)
        # self.log = self.log.append(df, ignore_index=True)
        self.log = pd.concat([self.log, df], ignore_index=True)

    def save(self):
        return self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log = pd.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))


def classification_accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def set_dataset_paths(args):
    args.train_path = 'data/file/%s.train.file' % args.dataset
    args.val_path = 'data/file/%s.val.file' % args.dataset
    args.test_path = 'data/file/%s.test.file' % args.dataset


def set_logger(filepath, resume_task_num=0):   # , mode='train'
    # global logger
    # Logger definition
    logger = logging.getLogger(__name__)
    # logging.basicConfig(filename=f'{hparams.saving_dir}/params.log', filemode='w', level=logging.INFO, format="%(message)s")   # a:续；w:重新

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()  # 控制台
    ch.setLevel(logging.INFO)
    mode = 'a' if resume_task_num > 0 else 'w'
    fh = logging.FileHandler(os.path.join(filepath, "params.log"), mode=mode)    # 控制台
    fh.setLevel(logging.DEBUG)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # logger = logging.getLogger('')
    # logger.setLevel(logging.INFO)
    # fh = logging.FileHandler(filepath)
    # fh.setLevel(logging.INFO)
    # ch = logging.StreamHandler(sys.stdout)
    # ch.setLevel(logging.INFO)
    #
    # _format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(_format)
    # ch.setFormatter(_format)
    #
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    return logger

def check_resume(hparams, resume_task_num=0):
    # save_folder = f'{hparams.saving_dir}/{task_num}_{task_id}'
    resume_folder = None
    # resume_task_num = 0
    # resume_from_epoch = -1

    if hparams.resume_task_num == 0 and resume_task_num == 0:
        pass
    else:
        # if hparams.resume_task_num != 0:
        #     resume_task_num = hparams.resume_task_num
        valid_files = glob.glob(f'{hparams.saving_dir}/{resume_task_num-1}_*')
        assert len(valid_files) == 1
        resume_folder = valid_files.pop()
        # resume_task_num = resume_task_num
        # print(resume_folder)
        return resume_folder, resume_task_num#, resume_from_epoch


    if glob.glob(f'{hparams.saving_dir}') and hparams.reuse_model:
        # valid_files = []
        for task_num in range(hparams.number_of_adpt-1, -1, -1):
            valid_files = glob.glob(f'{hparams.saving_dir}/{task_num}_*')
            if valid_files:
                assert len(valid_files) == 1
                resume_folder = valid_files.pop()
                resume_task_num = task_num
                break

        print(f'resume from task {resume_task_num}')
        # print(f'resume from task {resume_task_num}, epoch {resume_from_epoch}')
    # resume_folder = os.listdir(f'{hparams.saving_dir}/{task_num}_*')

    return resume_folder, resume_task_num#, resume_from_epoch


def load_checkpoint(model, resume_folder, hparams, task_num, backbone=False, before_retrain=False, adapter_task_id=None, max_bottleneck_size=None):   # max_bottleneck_size=None,
    device = torch.device(f"cuda:0")
    # checkpoint = torch.load(f'{resume_folder}/checkpoint-epoch={resume_from_epoch}.pth.tar')
    # valid_files = glob.glob(r"runs_E2E/SGD,TM19,TM20,MWOZ/cumulative-1_meta_mask_single_ADAPTER_EPC_1_LR_0.00625_BOTL_300_PERM_1_gpt2/0_[\'sgd_weather\']")
    # valid_files = os.listdir(f"{resume_folder}")
    resume_folder_ = ''
    for c in resume_folder: # 0_[\'sgd_weather\']中，需要对[]转义，否则会被误认为glob()中的[]标记
        if c == '[':
            c = '[[]'
        elif c == ']':
            c = '[]]'
        else:
            c = c
        resume_folder_ = resume_folder_ + c
    if hparams.retrain and task_num > 1 and not before_retrain:
        if hparams.split_mask and adapter_task_id == 1:
            valid_files = glob.glob(f'{resume_folder_}/model-epoch=*.pth.tar')
            print(f'task {task_num - 1} model loaded')
        else:
            valid_files = glob.glob(f'{resume_folder_}/model_retrain-epoch=*.pth.tar')
            print(f'task {task_num-1} model_retrain loaded')
    else:
        valid_files = glob.glob(f'{resume_folder_}/model-epoch=*.pth.tar')
        print(f'task {task_num-1} model loaded')
    assert len(valid_files) == 1
    checkpoint = torch.load(valid_files.pop())
    if hparams.CL == "ADAPTER":
        if max_bottleneck_size and max_bottleneck_size >= hparams.bottleneck_size:
            curr_model = model.model.adapter_blocks.state_dict()
            checkpoint_model = checkpoint['model_state_dict']
            for name, param in checkpoint_model.items():
                if 'efc1' in name:
                    # for i in range(task_num):
                    curr_model[name][:task_num, :curr_model[name].size(1)].fill_(-model.thres_emb)
                    # new_param[:new_param.size(0), :new_param.size(1)].copy_(param[:new_param.size(0), :new_param.size(1)])

                    # new_param = torch.Tensor(task_num, curr_model[name].size(1)).fill_(-np.inf)
                    # curr_model[name][:new_param.size(0), :new_param.size(1)].copy_(new_param)
                    # curr_model[name][:param.size(0), :param.size(1)].copy_(param[:param.size(0), :param.size(1)])
                    curr_model[name][:task_num, :param.size(1)].copy_(param[:task_num, :param.size(1)])
                    if hparams.todcl_mask:
                        curr_model[name][task_num, :param.size(1)].fill_(-model.thres_emb)

                elif len(param.size()) == 2:
                    curr_model[name][:param.size(0), :param.size(1)].copy_(param[:param.size(0), :param.size(1)])
                elif len(param.size()) == 1:
                    curr_model[name][:param.size(0)].copy_(param[:param.size(0)])
                else:
                    print()
        else:
            model.model.adapter_blocks.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.model.load_state_dict(checkpoint['model_state_dict'])
    # model = checkpoint['model_state_dict']
    # model.model.adapter_blocks.load_state_dict(checkpoint['model_state_dict'])  #
    model.task_list_seen = checkpoint['task_list_seen']
    model.lamb = checkpoint['lamb']
    # model.mask_pre = checkpoint['mask_pre']
    # model.mask_back = checkpoint['mask_back']
    # model.mask_pre_cumulative = checkpoint['mask_pre_cumulative']
    # model.mask_back_cumulative = checkpoint['mask_back_cumulative']

    if hparams.mask and not backbone:    #  and not backbone
        valid_files = glob.glob(f'{resume_folder_}/mask.pth.tar')
        assert len(valid_files) == 1
        checkpoint = torch.load(valid_files.pop())
        model.mask_pre = checkpoint['mask_pre']
        model.mask_back = checkpoint['mask_back']
        model.mask_pre_cumulative = checkpoint['mask_pre_cumulative']
        model.mask_back_cumulative = checkpoint['mask_back_cumulative']
        print(f'task {task_num-1} mask loaded')

        if max_bottleneck_size and max_bottleneck_size >= hparams.bottleneck_size:
            for t in range(task_num):
                mask = model.model.mask(t, s=model.smax)
                for key, value in mask.items():
                    mask[key] = torch.autograd.Variable(value.data.clone(), requires_grad=False)  # mask不再需要梯度

                if t == 0:
                    model.mask_pre = mask
                    if hparams.cumulative_mask:
                        model.mask_pre_cumulative = deepcopy(mask)  # deepcopy()
                else:
                    for key, value in model.mask_pre.items():
                        model.mask_pre[key] = torch.max(model.mask_pre[key], mask[key])
                        if hparams.cumulative_mask:
                            model.mask_pre_cumulative[key] = model.mask_pre_cumulative[key] + mask[key]

            # Weights mask
            model.mask_back = {}
            model.mask_back_cumulative = {}
            for n, p in model.model.named_parameters():
                vals = model.model.get_view_for(n, p, model.mask_pre)
                if hparams.cumulative_mask:
                    vals_cumulative = model.model.get_view_for(n, p, model.mask_pre_cumulative)
                if vals is not None:
                    model.mask_back[n] = 1 - vals
                    if hparams.cumulative_mask:
                        model.mask_back_cumulative[n] = vals_cumulative

            # for key, value in model.mask_back.items():
            #     if 'weight' in key:
            #         if 'fc1' in key:
            #             new_val = torch.Tensor(hparams.bottleneck_size, value.size(1)).fill_(1)
            #         elif 'fc2' in key:
            #             new_val = torch.Tensor(value.size(0), hparams.bottleneck_size).fill_(1)
            #         new_val[:value.size(0), :value.size(1)].copy_(value)
            #     elif 'bias' in key:
            #         if 'fc1' in key:
            #             new_val = torch.Tensor(hparams.bottleneck_size).fill_(1)
            #             new_val[:value.size(0)].copy_(value)
            #     else:
            #         print()
            #     model.mask_back[key] = new_val.to(device)

    if 'REPLAY' in hparams.CL and not backbone:
        valid_files = glob.glob(f'{resume_folder_}/reply_memory.pth.tar')
        assert len(valid_files) == 1
        checkpoint = torch.load(valid_files.pop())
        model.reply_memory = checkpoint['reply_memory']
        print(f'task {task_num-1} reply_memory loaded')

    elif hparams.retrain and not backbone:
        valid_files = glob.glob(f'{resume_folder_}/episodic_mem.pth.tar')
        assert len(valid_files) == 1
        checkpoint = torch.load(valid_files.pop())
        model.episodic_mem = checkpoint['episodic_mem']
        print(f'task {task_num-1} episodic_mem loaded')


def save_model(model, save_folder, best_epoch_idx, hparams, save_type='backbone', retrain=False):
    if save_type=='backbone':
        if retrain:
            save_folder_ = ''
            for c in save_folder:  # 0_[\'sgd_weather\']中，需要对[]转义，否则会被误认为glob()中的[]标记
                if c == '[':
                    c = '[[]'
                elif c == ']':
                    c = '[]]'
                else:
                    c = c
                save_folder_ = save_folder_ + c
            # if hparams.retrain:
            exist_files = glob.glob(f'{save_folder_}/model_retrain-epoch=*.pth.tar')
            if exist_files:
                assert len(exist_files)==1
                exist_file = exist_files.pop()
                os.remove(exist_file)   # shutil.rmtree(exist_file)
        else:
            if os.path.exists(save_folder):  # retrain要改！！
                shutil.rmtree(save_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            # os.mkdir(model_path)

        # filepath = self.args.checkpoint_format.format(save_folder=save_folder, epoch=epoch_idx + 1)
        # logger.info('saving current best model for epoch {}'.format(epoch + 1))
        if not retrain:
            model_path = os.path.join(save_folder, 'model-epoch={}.pth.tar'.format(best_epoch_idx))
        else:
            model_path = os.path.join(save_folder, 'model_retrain-epoch={}.pth.tar'.format(best_epoch_idx))

        # checkpoint = {'model_state_dict': model}
        if hparams.CL == "ADAPTER":
            model_state_dict = model.model.adapter_blocks.state_dict()
        else:
            model_state_dict = model.model.state_dict()

        checkpoint = {
            'model_state_dict': model_state_dict,  # model.module.state_dict()
            'task_list_seen': model.task_list_seen,
            'lamb': model.lamb,
            # 'mask_pre': model.mask_pre,
            # 'mask_back': model.mask_back,
            # 'mask_pre_cumulative': model.mask_pre_cumulative,
            # 'mask_back_cumulative': model.mask_back_cumulative,
        }
        torch.save(checkpoint, model_path)
        # model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.save_pretrained(model_path)

    elif save_type == 'mask':
        model_path = os.path.join(save_folder, 'mask.pth.tar')
        checkpoint = {
            'mask_pre': model.mask_pre,
            'mask_back': model.mask_back,
            'mask_pre_cumulative': model.mask_pre_cumulative,
            'mask_back_cumulative': model.mask_back_cumulative
        }
        torch.save(checkpoint, model_path)

    elif save_type == 'memory':
        if 'REPLAY' in hparams.CL:  # REPLAY/LIMIT-REPLAY
            model_path = os.path.join(save_folder, 'reply_memory.pth.tar')
            checkpoint = {
                'reply_memory': model.reply_memory
            }
            torch.save(checkpoint, model_path)
        else:
            model_path = os.path.join(save_folder, 'episodic_mem.pth.tar')
            checkpoint = {
                'episodic_mem': model.episodic_mem
            }
            torch.save(checkpoint, model_path)
    else:
        raise
