import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from random import sample
import pytorch_lightning as pl
from transformers import (AdamW, GPT2Tokenizer, GPT2LMHeadModel,T5Tokenizer, BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration)

from utils.dataloader import get_data_loaders, get_current_task_data, make_loader, make_val_loader
from collections import defaultdict

from utils.config import *
from fnmatch import fnmatch
from copy import deepcopy
from tqdm import tqdm

from transformers import logging
logging.set_verbosity_error()

from utils.utils_CL import calculate_task_adapter_id

# class Seq2SeqToD(pl.LightningModule):
class Seq2SeqToD(nn.Module):
    def __init__(self,args,load_dir=None):
        super().__init__()
        self.init_model(args)

        self.lr = args.lr
        self.current_task = 0
        self.fisher = defaultdict(list)
        self.optpar = defaultdict(list)
        # self.episodic_mem = defaultdict(list)
        self.CL = args.CL
        self.reg = args.reg
        self.first_task = True
        self.model_name = args.model_checkpoint
        self.reply_memory = []
        self.task_list_seen = []
        self.smax = 400
        self.thres_cosh = 50
        self.thres_emb = 6
        self.lamb = 0.75    # 控制loss中mask稀疏度的影响 0.75
        self.init_mask_mem()
        self.agem_mem_iter = None

    def init_mask_mem(self):
        self.mask_pre = None
        self.mask_back = None
        self.mask_pre_cumulative = None
        self.mask_back_cumulative = None
        self.mask_expand = None

        self.episodic_mem = defaultdict(list)
        # self.task_list_seen = []

    def init_model(self, args):
        if "t5" in args.model_checkpoint:
            model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "bart" in args.model_checkpoint:
            model = BartForConditionalGeneration.from_pretrained(args.model_checkpoint)
            tokenizer = BartTokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sep_token="[sep]")
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
        elif "gpt2" in args.model_checkpoint:
            if(args.CL == "ADAPTER"):
                if args.single:
                    from model.single_adapterGPT2 import GPT2Adapter
                else:
                    from model.adapterGPT2 import GPT2Adapter
                model = GPT2Adapter.from_pretrained(args.model_checkpoint)
                # model = GPT2Adapter.from_pretrained(args.model_checkpoint, cache_dir='/home/travisxu/mnt_file/ToDCL/download')
                model.add_adapters(args)
            elif args.CL == "LIMIT-REPLAY" and args.single:
                from model.single_adapterGPT2 import GPT2Adapter
                model = GPT2Adapter.from_pretrained(args.model_checkpoint)
                model.add_adapters(args)
            else:
                model = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
            # torch.set_printoptions(profile="full")
            tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint, bos_token="[bos]", eos_token="[eos]", sos_token="[SOS]", sep_token="[sep]",pad_token='[PAD]')
            model.resize_token_embeddings(new_num_tokens=len(tokenizer))
            # aa = dict(model.named_parameters())
        self.model = model
        self.tokenizer = tokenizer
        if USE_CUDA:
            self.model.cuda()
            if args.multi_gpu:
                self.model = nn.DataParallel(self.model)


    def set_number_of_tasks(self,n_tasks):
        self.n_tasks = n_tasks

    def set_up_gem(self):
        self.grad_dims = []
        for param in self.model.parameters():
            self.grad_dims.append(param.data.numel())
        dev = next(self.model.parameters()).device
        self.grads = torch.Tensor(sum(self.grad_dims), self.n_tasks).to(dev)

    def compute_PPL(self,batch,task_id=-1,device='cuda',s=None):
        with torch.no_grad():
            lm_logits, *_ = self.model(
                            input_ids=batch["input_id_PPL"].to(device),
                            attention_mask=None,
                            labels=None,
                            task_id=task_id,
                            s=s
                            )
        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = batch["output_id_PPL"].to(device)[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = torch.reshape(loss, shift_labels.size())
        return (loss.sum(1)/(loss!=0).sum(1)).tolist()

    def training_step(self, batch, batch_idx=None, s=None, masks=None, retrain=False):

        if self.CL == "GEM" and not self.first_task:
            dev = next(self.model.parameters()).device
            for id_task, (_,task_memory) in enumerate(self.episodic_mem.items()):
                batch_mem =  sample(task_memory,1)[0] # ==> we sample one batch from episodic memory
                self.model.zero_grad()
                (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                    attention_mask=batch_mem["attention_mask"].to(dev) if "gpt2" not in self.model_name else None,
                    labels=batch_mem["decoder_output"].to(dev)
                    )
                loss.backward()
                store_grad(self.model.parameters, self.grads, self.grad_dims, id_task)
            self.model.zero_grad()

        elif(self.CL == "AGEM" and not self.first_task):
            dev = next(self.model.parameters()).device

            try:
                batch_mem = next(self.agem_mem_iter)  # ==> we sample one batch from episodic memory
            except StopIteration:
                agem_data = []
                for mem_per_task in self.episodic_mem.values():
                    agem_data += mem_per_task
                self.agem_mem_iter = iter(make_loader(hparams, agem_data, self.tokenizer))
                # cur_task_loader = iter(self.agem_mem_loader)  # current_task_train_data
                batch_mem = next(self.agem_mem_iter)
            # batch_mem = sample(self.episodic_mem["all"],1)[0] # ==> we sample one batch from episodic memory
            self.model.zero_grad()
            (loss), *_ = self.model(input_ids=batch_mem["encoder_input"].to(dev),
                attention_mask=batch_mem["attention_mask"].to(dev) if "gpt2" not in self.model_name else None,
                labels=batch_mem["decoder_output"].to(dev)
                )
            loss.backward()
            grad_ref = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_ref.append(p.grad.view(-1))
            grad_ref = torch.cat(grad_ref) ## from eq. 10 of AGEM Paper

            self.model.zero_grad()

        # print(batch["encoder_input"].size())
        ## LOSS ON CURRENT DATA
        if(self.CL == "ADAPTER"):
            task_id = [self.task_list_seen.index(task_id) for task_id in batch["task_id"]] if retrain else self.task_list_seen.index(batch["task_id"][0])

            if s is None and masks is None:
                (loss), *_ = self.model(
                    input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"],
                    # task_id=len(self.task_list_seen),
                    task_id=self.task_list_seen.index(batch["task_id"][0]))
            else:
                (loss), *_ = self.model(
                    input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"],
                    task_id=task_id,
                    # task_id=len(self.task_list_seen)-1,
                    # task_id=self.task_list_seen.index(batch["task_id"][0]),
                    s=s, masks_pre=masks)

        else:
            (loss), *_ = self.model(input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"])

        if(self.CL == "AGEM" and not self.first_task):
            ## Code from https://github.com/GMvandeVen/continual-learning/blob/master/encoder.py#L244
            loss.backward()
            grad_cur = []
            for p in self.model.parameters():
                if p.requires_grad:
                    grad_cur.append(p.grad.view(-1))
            grad_cur = torch.cat(grad_cur)
            # -check inequality constrain
            angle = (grad_cur*grad_ref).sum()
            if angle < 0:
                # -if violated, project the gradient of the current batch onto the gradient of the replayed batch ...
                length_rep = (grad_ref*grad_ref).sum()
                grad_proj = grad_cur-(angle/length_rep)*grad_ref
                # -...and replace all the gradients within the model with this projected gradient
                index = 0
                for p in self.model.parameters():
                    if p.requires_grad:
                        n_param = p.numel()  # number of parameters in [p]
                        p.grad.copy_(grad_proj[index:index+n_param].view_as(p))
                        index += n_param
        elif self.CL == "GEM" and not self.first_task:
            loss.backward()
            store_grad(self.model.parameters, self.grads, self.grad_dims, id_task+1)
            indx = torch.LongTensor([j for j in range(id_task+1)])
            dotp = torch.mm(self.grads.to(dev)[:, id_task].unsqueeze(0), self.grads.to(dev).index_select(1, indx.to(dev)))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads.to(dev)[:, id_task].unsqueeze(1), self.grads.to(dev).index_select(1, indx.to(dev)), self.reg)
                # copy gradients back
                overwrite_grad(self.model.parameters, self.grads.to(dev)[:, id_task], self.grad_dims)

        elif self.CL == "L2" and not self.first_task:
            dev = next(self.model.parameters()).device
            l2_reg = 0

            for n,p in self.model.named_parameters():
                l = self.reg * (p - self.optpar[n].to(dev)).pow(2)
                l2_reg += l.sum()
            self.log('l2_reg', l2_reg, on_epoch=True)
            loss = loss + l2_reg
        elif self.CL == "EWC" and not self.first_task:
            dev = next(self.model.parameters()).device
            ewc_loss = 0
            for n,p in self.model.named_parameters():
                ## Eq (3) of https://arxiv.org/pdf/1612.00796.pdf
                # aa = (p - self.optpar[n].to(dev))
                # bb = aa.pow(2)
                # l = self.fisher[n].to(dev) * bb
                l = self.fisher[n].to(dev) * (p - self.optpar[n].to(dev)).pow(2)
                ewc_loss += l.sum()
            # self.log('EWC_reg', ewc_loss, on_epoch=True)
            # logger.info(f'EWC_reg: {ewc_loss}')
            loss = loss + self.reg * ewc_loss

        # self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, s=None, masks=None, val_retrain=False):
        if(self.CL == "ADAPTER"):
            task_id = [self.task_list_seen.index(task_id) for task_id in
                       batch["task_id"]] if val_retrain else self.task_list_seen.index(batch["task_id"][0])
            if s is None and masks is None:
                (loss), *_ = self.model(
                    input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"],
                    task_id=task_id)
            else:
                (loss), *_ = self.model(input_ids=batch["encoder_input"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["decoder_output"],
                                    task_id=task_id,
                                    s=s, masks_pre=masks)
        else:
            # print(batch["encoder_input"].size())
            (loss), *_ = self.model(input_ids=batch["encoder_input"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["decoder_output"]
                    )
        # self.log('val_loss', loss)
        return loss

    def backward(self, loss, optimizer=None, optimizer_idx=None):
        if (self.CL == "GEM" or self.CL == "AGEM") and not self.first_task:
            pass
        else:
            loss.backward()

    # travis
    def train_epoch(self, task_num=None, task_loader_length=None, pbar=None, optimizers=None, hparams=None, print_diag=False):
        self.model.train()
        loss = 0
        loss_reg = None
        t = task_num  # task_id=self.task_list_seen.index(batch["task_id"][0]

        for batch_idx, batch in pbar:
            try:
                if USE_CUDA:
                    batch["encoder_input"] = batch["encoder_input"].cuda()
                    batch["decoder_output"] = batch["decoder_output"].cuda()

                    # if batch_idx == 0 and print_diag:
                    #     print(batch["dial_id"])
                if hparams.mask:
                    s = (self.smax - 1 / self.smax) * batch_idx / task_loader_length + 1 / self.smax
                    if hparams.mask_CIL and t > 0:
                        loss = self.training_step(batch, batch_idx, s=s, masks=self.mask_pre)
                    else:
                        loss = self.training_step(batch, batch_idx, s=s)
                    # if batch_idx == 0 and print_diag:
                    #     logger.debug(f'loss: {loss}')
                    if not hparams.todcl_mask:
                        masks = self.model.mask(t, s)
                        loss_reg = self.hat_criterion_mask(masks, t)
                        loss += loss_reg
                    # if batch_idx == 0 and print_diag:
                    #     logger.debug(f'loss: {loss}')

                else:
                    loss = self.training_step(batch, batch_idx)
                    # loss = loss.mean()

                if hparams.gradient_accumulation_steps > 1:
                    loss = loss / hparams.gradient_accumulation_steps

                self.backward(loss)
                # model.backward(loss.mean())

                if (batch_idx + 1) % hparams.gradient_accumulation_steps == 0:
                    if hparams.mask:
                        # Restrict layer gradients in backprop
                        if self.mask_back is not None:  # t > 0
                            for n, p in self.model.named_parameters():
                                if n in self.mask_back:
                                    p.grad.data *= self.mask_back[n]

                        # if hparams.todcl_mask or hparams.expand_mask:
                        #     for n, p in self.model.named_parameters():
                        #         if n in self.mask_expand:
                        #             p.grad.data *= self.mask_expand[n].cuda()

                        # if not hparams.todcl_mask:
                        # Compensate embedding gradients
                        for n, p in self.model.named_parameters():
                            if p.grad is not None: # travis
                                if fnmatch(n, '*adapter_blocks.*.efc*'):    # 'adapter_blocks.0.efc1.weight'
                                # if 'adapter_mask.e' in n or n.startswith('e'):  # and (p.grad is not None)
                                    if hparams.expand_mask:
                                        num = torch.cosh(torch.clamp(s * p.data[:, :hparams.cur_bottleneck_size], -self.thres_cosh, self.thres_cosh)) + 1
                                        den = torch.cosh(p.data[:, :hparams.cur_bottleneck_size]) + 1
                                        p.grad.data[:, :hparams.cur_bottleneck_size] *= self.smax / s * num / den
                                    else:
                                        num = torch.cosh(torch.clamp(s * p.data, -self.thres_cosh, self.thres_cosh)) + 1
                                        den = torch.cosh(p.data) + 1
                                        p.grad.data *= self.smax / s * num / den

                        # lr_this_step = self.args.learning_rate * \
                        #                self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = lr_this_step

                    torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.max_norm)

                    optimizers.step()
                    # scheduler.step()
                    optimizers.zero_grad()

                    if hparams.mask:    # and not hparams.todcl_mask
                        # Constrain embeddings
                        for n, p in self.model.named_parameters():
                            if p.grad is not None:  # travis
                                if fnmatch(n, '*adapter_blocks.*.efc*'):  # 'adapter_blocks.0.efc1.weight'
                                    # if 'adapter_mask.e' in n or n.startswith('e'):
                                    if hparams.expand_mask:
                                        p.data[:, :hparams.cur_bottleneck_size] = torch.clamp(p.data[:, :hparams.cur_bottleneck_size], -self.thres_emb, self.thres_emb)
                                    else:
                                        p.data = torch.clamp(p.data, -self.thres_emb, self.thres_emb)

                        # if hparams.expand_mask:
                        #     self.model.reset_mask(hparams)

                    # if batch_idx < 8 and print_diag:
                    #     logger.debug(self.model.adapter_blocks[0].efc2.weight.data[1])

                loss = loss.item()*hparams.gradient_accumulation_steps
                description = 'LOSS:{:.3f}'.format(loss)
                # 'L:{:.2f},LG:{:.2f},LV:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                #                                                 print_loss_l)
                pbar.set_description(description)

            except RuntimeError as exception:
                raise exception

        if loss_reg:
            loss_reg = loss_reg.item()
        return loss, loss_reg

    def hat_criterion_mask(self, masks, t=None):
        reg = 0
        count = 0



        if self.mask_pre is not None:
            # for m,mp in zip(masks,self.mask_pre):
            for key in set(masks.keys()) & set(self.mask_pre.keys()):
                if hparams.expand_mask:
                    if 'fc1' in key:
                        assert t is not None
                        m = masks[key][:, :, :hparams.cur_bottleneck_size]
                        mp = self.mask_pre[key][:, :, :hparams.cur_bottleneck_size]
                        aux = 1 - mp
                        reg += (m * aux).sum()
                        count += aux.sum()
                else:
                    m = masks[key]
                    mp = self.mask_pre[key]
                    aux = 1 - mp
                    reg += (m * aux).sum()
                    count += aux.sum()
        else:
            for m_key, m_value in masks.items():
                # reg += m_value.sum()
                if hparams.expand_mask: #  and 'fc1' in m_key
                    if 'fc1' in m_key:
                        assert t is not None
                        reg += m_value.sum()
                        count += np.prod(m_value.size()).item() / hparams.bottleneck_size * hparams.cur_bottleneck_size
                else:
                    reg += m_value.sum()
                    count += np.prod(m_value.size()).item()

        reg /= count

        return self.lamb * reg  # self.ce(outputs, targets) +

    def eval_epoch(self, t=None, pbar=None, hparams=None, trained_task=None):
        self.model.eval()
        total_loss = 0
        total_num = 0
        val_retrain = True if hparams.val_retrain and t > 0 else False
        try:
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    if USE_CUDA:
                        batch["encoder_input"] = batch["encoder_input"].cuda()
                        batch["decoder_output"] = batch["decoder_output"].cuda()

                    if hparams.mask:
                        s = self.smax
                        if hparams.mask_CIL and t > 0:
                            loss = self.validation_step(batch, batch_idx, s=s, masks=self.mask_pre)
                        else:
                            loss = self.validation_step(batch, batch_idx, s=s, val_retrain=val_retrain)
                        masks = self.model.mask(t, s)
                        if not hparams.todcl_mask:
                            loss += self.hat_criterion_mask(masks, t)
                    else:
                        loss = self.validation_step(batch, batch_idx)

                    real_b = batch["encoder_input"].size(0)
                    total_loss += loss.data.cpu().numpy().item()*real_b
                    total_num += real_b

                # epoch_mean_loss = total_loss / len(val_loader[task_id])
                epoch_mean_loss = total_loss / total_num

                return epoch_mean_loss

        except RuntimeError as exception:
            raise exception

    def retrain_epoch(self, task_num, task_loader_length, pbar, optimizers, hparams):#, masks=None):
        self.model.train()
        t = task_num  # task_id=self.task_list_seen.index(batch["task_id"][0]
        s = self.smax

        for batch_idx, batch in pbar:
            try:
                if USE_CUDA:
                    batch["encoder_input"] = batch["encoder_input"].cuda()
                    batch["decoder_output"] = batch["decoder_output"].cuda()

                assert hparams.mask
                loss = self.training_step(batch, batch_idx, s=s, retrain=True)     # masks=self.mask_pre

                if hparams.retrain_gradient_accumulation_steps > 1:
                    loss = loss / hparams.retrain_gradient_accumulation_steps

                self.backward(loss)
                # model.backward(loss.mean())

                if (batch_idx + 1) % hparams.retrain_gradient_accumulation_steps == 0:
                    self.restrict_retrain_grad(t, hparams)

                    torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.max_norm)
                    optimizers.step()
                    # scheduler.step()
                    optimizers.zero_grad()

                description = 'LOSS:{:.3f}'.format(loss.item()*hparams.retrain_gradient_accumulation_steps)
                # 'L:{:.2f},LG:{:.2f},LV:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v,
                #                                                 print_loss_l)
                pbar.set_description(description)

            except RuntimeError as exception:
                raise exception

    def meta_train_epoch(self, task_num, task_loader_length, pbar, optimizers, hparams, cur_task_loader=None, task_loader=None):#, masks=None):
        # train_epoch
        self.model.train()
        t = task_num  # task_id=self.task_list_seen.index(batch["task_id"][0]
        s = self.smax

        total_meta_step = 0
        finished_directions = 0
        # query_step = 0
        query_loss = 0
        losses_meta = []
        current_direction_step = True
        init_state = {k: v.cpu() for k, v in self.model.adapter_blocks.state_dict().items()}
        # init_state = deepcopy(self.model.adapter_blocks.state_dict())

        # task_num = len(self.tasks)
        if hparams.split_mask:
            adapter_id, adapter_task_id = calculate_task_adapter_id(task_num)
            coef_old = adapter_task_id / (adapter_task_id + 1)
            coef_new = 1 / (adapter_task_id + 1)
        else:
            coef_old = task_num / (task_num + 1)
            coef_new = 1 / (task_num + 1)
        # loss += coef_new * loss_new_balance + coef_old * loss_old_balance

        for batch_idx, batch in pbar:
            try:
                assert hparams.mask
                if current_direction_step:
                    try:
                        batch_cur = next(cur_task_loader)
                    except StopIteration:
                        cur_task_loader = iter(task_loader)  # current_task_train_data
                        batch_cur = next(cur_task_loader)
                    if USE_CUDA:
                        batch_cur["encoder_input"] = batch_cur["encoder_input"].cuda()
                        batch_cur["decoder_output"] = batch_cur["decoder_output"].cuda()
                    for _ in range(hparams.fast_update):  # meta_steps for current direction
                        loss = self.training_step(batch_cur, s=s)
                        # loss = self.training_step(batch, batch_idx, masks=self.mask_pre)
                        self.backward(loss)
                        self.restrict_retrain_grad(t, hparams)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.max_norm)
                        # self.model.do_weight_decay_and_make_grads_zero()
                        # if self.model.piggymask_optimizer is not None:
                        #     self.model.piggymask_optimizer.step()
                        #     self.model.piggymask_optimizer.zero_grad()
                        optimizers.step()

                    # current_direction_step = False
                    total_meta_step += 1
                    # continue

                    for query_step in range(hparams.meta_query_step):
                        try:
                            batch_cur = next(cur_task_loader)
                        except StopIteration:
                            cur_task_loader = iter(task_loader)  # current_task_train_data
                            batch_cur = next(cur_task_loader)
                        if USE_CUDA:
                            batch_cur["encoder_input"] = batch_cur["encoder_input"].cuda()
                            batch_cur["decoder_output"] = batch_cur["decoder_output"].cuda()
                        loss = self.training_step(batch_cur, s=s)
                        # loss = self.training_step(batch, batch_idx, masks=self.mask_pre)
                        if hparams.balance:
                            loss *= coef_new
                        query_loss += loss/hparams.meta_query_step
                        # if batch_idx == 0:
                        #     logger.debug(f"loss_new: {query_loss}")
                        # query_step += 1
                        total_meta_step += 1

                    if USE_CUDA:
                        batch["encoder_input"] = batch["encoder_input"].cuda()
                        batch["decoder_output"] = batch["decoder_output"].cuda()
                        loss = self.training_step(batch, s=s, retrain=True)
                        if hparams.balance:
                            loss *= coef_old
                        # if batch_idx == 0:
                        #     logger.debug(f"loss_old: {loss}")
                        query_loss += loss

                    # if query_step == hparams.meta_query_step:
                    finished_directions += 1
                    losses_meta.append(query_loss)
                    query_loss = 0
                    # query_step = 0
                    self.model.adapter_blocks.load_state_dict(init_state)
                    optimizers.zero_grad()
                    # current_direction_step = True

                if total_meta_step % (hparams.direction * (hparams.meta_query_step + 1)) == 0:
                    total_meta_step = 0
                    finished_directions = 0
                    # query_step = 0
                    # current_direction_step = True
                    if losses_meta:
                        self.model.adapter_blocks.load_state_dict(init_state)
                        optimizers.zero_grad()
                        loss_meta = torch.stack(losses_meta).sum(0) / hparams.direction
                        self.backward(loss_meta)
                        self.restrict_retrain_grad(t, hparams)
                        torch.nn.utils.clip_grad_norm_(self.parameters(), hparams.max_norm)
                        # if self.model.piggymask_optimizer is not None:
                        #     self.model.piggymask_optimizer.step()
                        #     self.model.piggymask_optimizer.zero_grad()
                        optimizers.step()
                        optimizers.zero_grad()
                    losses_meta = []
                    # init_state = deepcopy(self.model.adapter_blocks.state_dict())
                    init_state = {k: v.cpu() for k, v in self.model.adapter_blocks.state_dict().items()}
                    # if len(pbar) - i < args['direction'] * (args['meta_query_step'] + 1):
                    #     if self.model.piggymask_optimizer is not None:
                    #         self.model.piggymask_scheduler.step()
                    #     break

            except RuntimeError as exception:
                raise exception
        return cur_task_loader

    def eval_retrain(self, t, pbar, hparams, trained_task=None):
        self.model.eval()
        s = self.smax
        total_loss = 0
        total_num = 0
        try:
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    if USE_CUDA:
                        batch["encoder_input"] = batch["encoder_input"].cuda()
                        batch["decoder_output"] = batch["decoder_output"].cuda()

                    assert hparams.mask
                    val_retrain = True if hparams.val_retrain else False
                    # loss = self.validation_step(batch, batch_idx, s=s, retrain=retrain)   # , masks=self.mask_pre
                    # if hparams.mask:
                    # s = self.smax
                    if hparams.mask_CIL and t > 0:
                        loss = self.validation_step(batch, batch_idx, s=s, masks=self.mask_pre)
                    else:
                        loss = self.validation_step(batch, batch_idx, s=s, val_retrain=val_retrain)
                    masks = self.model.mask(t, s)
                    loss += self.hat_criterion_mask(masks, t)

                    real_b = batch["encoder_input"].size(0)
                    total_loss += loss.data.cpu().numpy().item()*real_b
                    total_num += real_b

                epoch_mean_loss = total_loss / total_num
                return epoch_mean_loss

        except RuntimeError as exception:
            raise exception

    def restrict_retrain_grad(self, t, hparams):
        if hparams.mask:
            # Restrict layer gradients in backprop
            if self.mask_back is not None:  # t > 0
                for n, p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data *= (1 - self.mask_back[n])
                        # if hparams.expand_mask:
                        #     p.grad.data *= self.mask_expand[n].cuda()
                        if hparams.cumulative_mask:
                            # threshold = self.mask_back_cumulative[n].gt(len(self.task_list_seen) - 1)
                            # p.grad.data[threshold] = 0
                            p.grad.data *= self.mask_back_cumulative[n]/len(self.task_list_seen)
                            # p.grad.data *= (self.mask_back_cumulative[n])

    def sampling(self, train_datasets, task_id, task_num=-1):
        self.model.train()
        device = torch.device(f"cuda:0")
        if hparams.split_mask:
            adapter_id, adapter_task_id = calculate_task_adapter_id(task_num)
            size_per_task = hparams.episodic_mem_size // (adapter_task_id+1)
        else:
            size_per_task = hparams.episodic_mem_size // len(self.task_list_seen)
        s = self.smax

        for mem_task_id in self.task_list_seen:
            if mem_task_id == task_id:
                sample_datasets = train_datasets[mem_task_id]
            elif mem_task_id in self.episodic_mem.keys():
                sample_datasets = self.episodic_mem[mem_task_id]
            else:
                continue

            if not hparams.uncertainty:
                self.episodic_mem[mem_task_id] = sample(train_datasets[task_id], min(len(sample_datasets), size_per_task))
            else:   # uncertainty_sampling
                mem_task_loader = make_val_loader(hparams, sample_datasets, self.tokenizer)
                perplexity_list, sample_index = [], []
                t = self.task_list_seen.index(mem_task_id)
                for i in range(hparams.augmentation):
                    perplexity_per_aug = []
                    # for idx_b, batch in tqdm(enumerate(mem_task_loader), total=len(mem_task_loader)):
                    for idx_b, batch in enumerate(mem_task_loader):
                        # ppl_batch = self.compute_PPL(batch, task_id=t, device=device)  ## one value per batch
                        with torch.no_grad():
                            lm_logits, *_ = self.model(
                                input_ids=batch["encoder_input"].to(device),
                                # input_ids=batch["encoder_input"],
                                attention_mask=None,
                                labels=None,
                                task_id=t,    # task_id,
                                s=s
                                # masks_pre=self.mask_pre
                            )
                        # Shift so that tokens < n predict n
                        shift_logits = lm_logits[..., :-1, :].contiguous()
                        shift_labels = batch["reply_output"].to(device)[..., 1:].contiguous() # decoder_output
                        # Flatten the tokens
                        loss_fct = CrossEntropyLoss(reduction='none')
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        loss = torch.reshape(loss, shift_labels.size())
                        # return (loss.sum(1) / (loss != 0).sum(1)).tolist()
                        ppl_batch = (loss.sum(1) / (loss != 0).sum(1)).tolist()
                        perplexity_per_aug += ppl_batch
                    perplexity_list.append(perplexity_per_aug)

                perplexity_list = [sum(e) / len(e) for e in zip(*perplexity_list)]

                sorted_id = sorted(range(len(perplexity_list)), key=lambda k: perplexity_list[k], reverse=True)
                jump_idx = len(sample_datasets) // size_per_task
                sample_index = sorted_id[::jump_idx][:size_per_task]
                self.episodic_mem[mem_task_id] = [sample_datasets[i] for i in sample_index]
                logger.debug([self.episodic_mem[mem_task_id][i]['dial_id'] for i in range(min(10, len(self.episodic_mem[mem_task_id])))])