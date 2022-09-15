import numpy as np
# import quadprog
import torch
from utils.config import *
from transformers import AdamW
from utils import Optimizers
from copy import deepcopy

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


def calculate_mask_ratio(mask):
    total_elem = 0
    curr_task_elem = 0
    # reg = 0

    for m_key, m_value in mask.items():
        # reg += m_value.sum().item()

        if hparams.expand_mask:
            if 'fc1' in m_key:
                # assert t is not None
                curr_task_elem += torch.sum(m_value.gt(0.5)).cpu()
                total_elem += np.prod(m_value.size()).item() / hparams.bottleneck_size * hparams.cur_bottleneck_size
        else:
            curr_task_elem += torch.sum(m_value.gt(0.5)).cpu()
            total_elem += np.prod(m_value.size()).item()
        # total_elem += m_value.numel() # 同上

    return float(curr_task_elem) / total_elem

def calculate_mask(model, t):
    # t = task_num
    # if model.args.multi_gpu and not model.args.distributed:
    #     mask = model.model.module.mask(t, s=model.smax)
    # else:
    #     mask = model.model.mask(t, s=model.smax)
    if hparams.split_mask:
        adapter_id, adapter_task_id = calculate_task_adapter_id(t)
        appendix = f'.mixadaptermask.{adapter_id}'
    else:
        appendix = ''

    mask = model.model.mask(t, s=model.smax, appendix=appendix)
    for key, value in mask.items():
        mask[key] = torch.autograd.Variable(value.data.clone(), requires_grad=False)  # mask不再需要梯度

    curr_mask_ratio = calculate_mask_ratio(mask)
    logger.info('Current Mask Ratio:{:.2f}'.format(curr_mask_ratio))

    if t == 0 or model.mask_pre is None:
        model.mask_pre = mask
        if hparams.cumulative_mask:
            model.mask_pre_cumulative = deepcopy(mask)  # deepcopy()
    else:
        for key, value in model.mask_pre.items():
            model.mask_pre[key] = torch.max(model.mask_pre[key], mask[key])
            if hparams.cumulative_mask:
                model.mask_pre_cumulative[key] = model.mask_pre_cumulative[key] + mask[key]

        total_mask_ratio = calculate_mask_ratio(model.mask_pre)
        logger.info('Total Mask Ratio:{:.2f}'.format(total_mask_ratio))

    # Weights mask
    model.mask_back = {}
    model.mask_back_cumulative = {}
    aa = []
    for n, p in model.model.named_parameters():
        aa.append(n)
        # if model.args.multi_gpu and not model.args.distributed:
        #     vals = model.model.module.get_view_for(n, p, model.mask_pre)
        # else:
        #     vals = model.model.get_view_for(n, p, model.mask_pre)
        vals = model.model.get_view_for(n, p, model.mask_pre, appendix=appendix)
        if hparams.cumulative_mask:
            vals_cumulative = model.model.get_view_for(n, p, model.mask_pre_cumulative, appendix=appendix)
        if vals is not None:
            model.mask_back[n] = 1 - vals
            if hparams.cumulative_mask:
                model.mask_back_cumulative[n] = vals_cumulative

    return mask


def calculate_mask_expand(model, task_num):
    masks = {}
    for layer_id in range(model.model.config.n_layer):
        fc1_key = 'adapter_blocks.' + str(layer_id) + '.fc1'  # gfc1
        fc2_key = 'adapter_blocks.' + str(layer_id) + '.fc2'  # gfc2
        masks[fc1_key] = torch.ones(hparams.bottleneck_size)
        masks[fc1_key][:(task_num + 1)*50].fill_(0)
        masks[fc2_key] = torch.zeros(model.model.config.n_embd)
        # masks[fc1_key], masks[fc2_key] = model.model.adapter_blocks[layer_id].mask(t, s)

    model.mask_expand = {}
    for n, p in model.model.named_parameters():
        for layer_id in range(model.model.config.n_layer):
            vals = None
            if n == 'adapter_blocks.' + str(layer_id) + '.fc1.weight':
                vals = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
            elif n == 'adapter_blocks.' + str(layer_id) + '.fc1.bias':
                vals = masks[n.replace('.bias', '')].data.view(-1)
            elif n == 'adapter_blocks.' + str(layer_id) + '.fc2.weight':
                post = masks[n.replace('.weight', '')].data.view(-1, 1).expand_as(p)
                pre = masks[n.replace('.weight', '').replace('fc2', 'fc1')].data.view(1, -1).expand_as(p)
                vals = torch.max(post, pre) # torch.min(post, pre)
            elif n == 'adapter_blocks.' + str(layer_id) + '.fc2.bias':
                vals = masks[n.replace('.bias', '')].data.view(-1)
            if vals is not None:
                model.mask_expand[n] = 1 - vals
    # model.mask_expand = torch.full(hparams.bottleneck_size, np.inf)


def calculate_bottleneck_size(task_num):
    if hparams.split_mask:
        # if hparams.num_of_adapter == 3:
        #     adapter_id = calculate_task_adapter_id(task_num)[0]
        #     cur_bottleneck_size = 300 if adapter_id != 2 else 50
        # else:
        raise
    else:
        mask_task_id = None
        if hparams.num_of_mask is None:
            mask_id = task_num
            cur_bottleneck_size = (mask_id+1)*50
        elif hparams.num_of_mask == 6:
            mask_id = abs(task_num-1)//2
            cur_bottleneck_size = (mask_id+1)*100
        elif hparams.num_of_mask == 4:
            mask_id = abs(task_num - 1) // 3
            cur_bottleneck_size = (mask_id+1)*150
        elif hparams.num_of_mask == 3:
            mask_id = abs(task_num - 1) // 4
            # if hparams.split_mask:
            #     mask_task_id = (task_num - 1) % 4 if mask_id > 0 else task_num
            cur_bottleneck_size = (mask_id+1)*200
        elif hparams.num_of_mask == 2:
            mask_id = abs(task_num - 1) // 6
            # if hparams.split_mask:
            #     mask_task_id = (task_num - 1) % 6 if mask_id > 0 else task_num
            cur_bottleneck_size = (mask_id + 1) * 300
        else:
            raise
    print(f"cur_bottleneck_size: {cur_bottleneck_size}")
    return cur_bottleneck_size

def calculate_task_adapter_id(task_num):
    adapter_id, adapter_task_id = None, None
    # if hparams.num_of_adapter == 3:
    for adapter_id in range(hparams.num_of_adapter):
        if sum(hparams.task_adapter_list[:(adapter_id+1)]) > task_num:
            adapter_task_id = task_num - sum(hparams.task_adapter_list[:adapter_id]) if adapter_id > 0 else task_num
            break
        else:
            continue
        # if hparams.bottleneck_size == 600:
        #     adapter_id = abs(task_num - 1) // 4
        #     adapter_task_id = (task_num - 1) % 4 if adapter_id > 0 else task_num
        # elif hparams.bottleneck_size == 650:
        #     adapter_id = task_num // 6
        #     adapter_task_id = task_num % 6
        # adapter_id = abs(task_num - 1) // 4
        # adapter_task_id = (task_num - 1) % 4 if adapter_id > 0 else task_num
    # elif hparams.num_of_adapter == 2:
        # adapter_id = 0 if task_num == 0 else 1
        # adapter_task_id = (task_num - 1) if adapter_id > 0 else task_num
        # adapter_id = abs(task_num - 1) // 6
        # adapter_task_id = (task_num - 1) % 6 if adapter_id > 0 else task_num
    # else:
    #     raise
    return adapter_id, adapter_task_id

def calculate_adapter_num(num_of_adapter):
    if num_of_adapter == 3:
        # if hparams.bottleneck_size == 600:
        if "TM" in hparams.dataset_list:
            if hparams.bottleneck_size == 600:
                hparams.task_adapter_list = [5, 4, 4]
                hparams.bottleneck_size_list = [200, 200, 200]
            elif hparams.bottleneck_size == 650:
                hparams.task_adapter_list = [6,6,1]
                hparams.bottleneck_size_list = [300,300,50]
            elif hparams.bottleneck_size == 900:
                hparams.task_adapter_list = [5,4,4]
                hparams.bottleneck_size_list = [300,300,300]
        elif "SGD" in hparams.dataset_list:
            if hparams.begin_domain == 13:
                hparams.task_adapter_list = [5, 4, 4]
                hparams.bottleneck_size_list = [200, 200, 200]
            elif hparams.begin_domain == 12:
                hparams.task_adapter_list = [6, 5, 5]
                hparams.bottleneck_size_list = [300, 300, 300]
            elif hparams.begin_domain == 7:
                hparams.task_adapter_list = [7, 6, 5]
                hparams.bottleneck_size_list = [300, 300, 300]
            else:
                raise
        else:
            raise
    elif num_of_adapter == 2:
        if "SGD" in hparams.dataset_list and "TM" in hparams.dataset_list:
            pass
        elif "SGD" in hparams.dataset_list:
            hparams.task_adapter_list = [7, 6]  # [7, 6] [1,2]
            hparams.bottleneck_size_list = [300, 300]
        elif "TM" in hparams.dataset_list:
            hparams.task_adapter_list = [7, 6]  # [7, 6] [1,2]
            hparams.bottleneck_size_list = [300, 300]
    else:
        raise

    if not hparams.mode == 'test':
        logger.info(f"task_adapter_list {hparams.task_adapter_list}, bottleneck_size_list {hparams.bottleneck_size_list}")

    # hparams.bottleneck_size = int(hparams.bottleneck_size / hparams.num_of_adapter)
    # logger.info(f'bottleneck size per adapter: {hparams.bottleneck_size}')
    # return adapter_num

def set_requires_grad(model, retrain=False):
    # assert hparams.CL == "ADAPTER" or hparams.CL == "LIMIT-REPLAY"
    if retrain:
        # print('train adapter w/o efc!')    # transformer+adapter
        for n, p in model.named_parameters():
            if 'transformer' in n:
                p.requires_grad = False # True/False
            elif 'adapter_blocks' in n:
                if '.efc' in n:
                    p.requires_grad = False
                    # p.requires_grad = True if '.efc1' in n and hparams.expand_mask else False
                else:
                    p.requires_grad = True # True/False
    else:
        # if model.CL == "LIMIT-REPLAY" and not hparams.single:
        #     print('Train transformer!') # adapter+
        # elif hparams.freeze_emb:
        #     print('Train adapter+transformer w/o emb!')
        # else:
        #     print('Train adapter!')   # transformer+
        for n, p in model.named_parameters():   # lnf.weight/bias
            if model.CL == "ADAPTER" or hparams.single:
                if 'transformer' in n:
                    p.requires_grad = False # True/False
                    # if hparams.freeze_emb and ('wte' in n or 'wpe' in n):
                    #     p.requires_grad = False
                    # else:
                    #     p.requires_grad = True
                elif 'adapter_blocks' in n:
                    if hparams.expand_mask: # or hparams.split_mask
                        p.requires_grad = False if '.efc2' in n else True
                    elif hparams.todcl_mask:
                        p.requires_grad = False if '.efc' in n else True
                    else:
                        p.requires_grad = True
            else:
                p.requires_grad = True

            # if model.CL != "LIMIT-REPLAY" and model.CL != "ADAPTER":
            #     p.requires_grad = True
            # elif model.CL == "LIMIT-REPLAY" and not hparams.single:
            #     p.requires_grad = True

def configure_optimizers(model, retrain=False):
    if(model.CL=="ADAPTER"):
        optimizers = Optimizers()
        lr = hparams.retrain_lr_factor * model.lr if retrain else model.lr
        if not retrain:
            if hparams.expand_mask: #  or hparams.split_mask
                freeze_layers = ["transformer", '.efc2']
                parameters_to_update = [p for n, p in model.named_parameters() if
                                        not (True in [ele in str(n) for ele in freeze_layers])]
                logger.info(f'train adapter w/o efc2 with lr {lr}')
            elif hparams.todcl_mask:
                freeze_layers = ["transformer", '.efc']
                parameters_to_update = [p for n, p in model.named_parameters() if
                                        not (True in [ele in str(n) for ele in freeze_layers])]
                logger.info(f'train adapter w/o efc with lr {lr}')
            else:
                # if hparams.mask_CIL:
                # lr_adapters = model.lr
                # lr_transformer = 0.01*model.lr
                params_to_optimize_via_AdamW = [p for n, p in model.named_parameters() if "adapter" in str(n)]
                '''if hparams.freeze_emb:
                        freeze_layers = ["adapter", '.wte.', '.wpe.']
                        params_to_optimize_via_AdamW2 = [p for n, p in model.named_parameters() if not (True in [ele in str(n) for ele in freeze_layers])]
                    else:
                        params_to_optimize_via_AdamW2 = [p for n, p in model.named_parameters() if "adapter" not in str(n)]        
                    optimizer_network = AdamW(params_to_optimize_via_AdamW2, lr=lr_transformer, correct_bias=True)  # weight_decay=0.0                
                    optimizers.add(optimizer_network, lr_transformer)'''

                parameters_to_update = params_to_optimize_via_AdamW
                logger.info('Train adapter!')

        else:
            freeze_layers = ["transformer", '.efc']
            parameters_to_update = [p for n, p in model.named_parameters() if
                                             not (True in [ele in str(n) for ele in freeze_layers])]
            logger.info(f'train adapter w/o efc with lr {lr}')
            # parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)]
            # retrain_lr = hparams.retrain_lr_factor * model.lr
            # optimizer = AdamW(parameters_to_update, lr=lr, correct_bias=True)
            # optimizers.add(optimizer, lr)


        optimizer = AdamW(parameters_to_update, lr=lr, correct_bias=True)  # weight_decay=1e-8
        optimizers.add(optimizer, lr)
        optimizers.zero_grad()
        return optimizers

    elif hparams.single:    # (model.CL=="LIMIT-REPLAY") and
        parameters_to_update = [p for n, p in model.named_parameters() if "adapter" in str(n)]
        return AdamW(parameters_to_update, lr=model.lr, correct_bias=True)
    else:
        print('Train transformer!')
        return AdamW(model.parameters(), lr=model.lr, correct_bias=True)
