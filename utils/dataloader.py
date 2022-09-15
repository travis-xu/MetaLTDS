import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from functools import partial
from utils.preprocess import get_datasets
from collections import defaultdict
import pprint
import random

from tabulate import tabulate

from utils.config import *
from copy import deepcopy
from random import sample

pp = pprint.PrettyPrinter(indent=4)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data, domains=None):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item

    def __len__(self):
        return self.dataset_len

def get_intent_from_dial(args,data,task_id,tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                pass
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif((t['spk'] == "API") and idx_t!=0 and t["utt"]!= ""):
                intent = " ".join(t["service"])
                dialogues.append({"history":" ".join(plain_history[-args.max_history:]),
                                  "reply":f'{intent} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:])+ f'[SOS]{intent} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":t["turn_id"],
                                  "task_id":task_id})
    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues

def get_DST_from_dial(args,data,task_id,tokenizer):
    dialogues = []
    for dial in data:
        plain_history = []
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                pass
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif((t['spk'] == "API") and idx_t!=0 and t["utt"]!= ""):
                slots = t["utt"].strip()
                dialogues.append({"history":" ".join(plain_history[-args.max_history:]),
                                  "reply":f'{slots} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:])+ f'[SOS]{slots} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":t["turn_id"],
                                  "task_id":task_id})
    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_NLG_from_dial(args,data,task_id,tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                latest_API_OUT = f"{t['utt'].strip()}"
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                if(latest_API_OUT != ""):
                    dialogues.append({"history":latest_API_OUT,
                                    "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                    "history_reply": latest_API_OUT + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                    "spk":t["spk"],
                                    "dataset":t["dataset"],
                                    "dial_id":t["id"],
                                    "turn_id":t["turn_id"],
                                    "task_id":task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = ""

    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues

def get_e2e_from_dial(args,data,task_id,tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9" and f'{t["id"]}' == "dlg-fdd242eb-56be-48c0-a56e-5478472500d0":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                latest_API_OUT = f"{t['spk']}: {t['utt'].strip()}"
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                dialogues.append({"history":" ".join(plain_history[-args.max_history:] + [latest_API_OUT]),
                                  "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:] + [latest_API_OUT])+ f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":t["turn_id"],
                                  "task_id":task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = "API-OUT: "
            elif((t['spk'] == "API") and idx_t!=0 and t["utt"]!= ""):
                dialogues.append({"history":" ".join(plain_history[-args.max_history:]),
                                  "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:])+ f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":str(t["turn_id"])+"API",
                                  "task_id":task_id})
    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_current_task_data(args,dataset_dic,task_id,number_of_sample):
    temp_aug = random.sample(dataset_dic,min(number_of_sample,len(dataset_dic)))
    aug_data = []
    cnt_API = 0
    for d in temp_aug:
        ## add a first token for the generation
        if(args.task_type=="E2E"):
            if(d["spk"]=="API"):
                cnt_API += 1
                d["history_reply"] = f"[{str(eval(task_id)[0])}-API]"+d["history_reply"]
            else:
                d["history_reply"] = f"[{str(eval(task_id)[0])}]"+d["history_reply"]
        else:
            d["history_reply"] = f"[{str(eval(task_id)[0])}]"+d["history_reply"]
        aug_data.append(d)
    return aug_data

def _cuda(x):
    if USE_CUDA:
        return x#.cuda()
    else:
        return x

def collate_fn(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False)
    batch_data["encoder_input"] = _cuda(input_batch["input_ids"])
    batch_data["attention_mask"] = _cuda(input_batch["attention_mask"])
    output_batch = tokenizer(batch_data["reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = _cuda(output_batch['input_ids'])
    return batch_data

def collate_fn_GPT2(data,tokenizer):
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False,return_attention_mask=False)
    batch_data["encoder_input"] = _cuda(input_batch["input_ids"])
    batch_data["attention_mask"] = None
    output_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = _cuda(output_batch['input_ids'])

    #### DATA FOR COMPUTING PERPLEXITY OF DIALOGUE HISTORY ==> FOR ADAPTER
    batched_history = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False,return_attention_mask=False)
    batch_data["input_id_PPL"] = _cuda(batched_history['input_ids'])
    batched_history_out = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False, return_attention_mask=False)
    batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["output_id_PPL"] = _cuda(batched_history_out['input_ids']) ### basically just remove pad from ppl calculation

    ### remove pad and history
    if hparams.CL != "LAMOL":   # LAMOL的生成数据套用了初始数据格式，而只改变了其中的history_reply
        batch_data["reply_output"] = deepcopy(batch_data["decoder_output"])
        batch_data["reply_output"][:, :batch_data['input_id_PPL'].size(-1)].masked_fill_(batch_data['input_id_PPL']!=tokenizer.pad_token_id, -100)
        batch_data["reply_output"] = _cuda(batch_data["reply_output"])

    return batch_data


def make_loader(args,list_sample,tokenizer):
    collate_fn_ = collate_fn_GPT2 if("gpt2" in args.model_checkpoint) else collate_fn
    return DataLoader(DatasetTrain(list_sample), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))

def make_val_loader(args,list_sample,tokenizer):
    collate_fn_ = collate_fn_GPT2 if("gpt2" in args.model_checkpoint) else collate_fn
    return DataLoader(DatasetTrain(list_sample), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))


def get_data_loaders(args, tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list=args.dataset_list.split(','),setting=args.setting,verbose=args.verbose,develop=args.debug)

    collate_fn_ = collate_fn_GPT2 if("gpt2" in args.model_checkpoint) else collate_fn
    if(test):
        datasets = {"test":{}}
    else:
        datasets = {"train":{}, "dev": {}, "test":{}}

    for split in datasets.keys():
        for tasks_id, task in aggregate["BYDOMAIN"][split].items():
            if(args.task_type == "E2E"):
                datasets[split][tasks_id] = get_e2e_from_dial(args,task,tasks_id,tokenizer)
            elif(args.task_type == "INTENT"):
                datasets[split][tasks_id] = get_intent_from_dial(args,task,tasks_id,tokenizer)
            elif(args.task_type == "DST"):
                datasets[split][tasks_id] = get_DST_from_dial(args,task,tasks_id,tokenizer)
            elif(args.task_type == "NLG"):
                datasets[split][tasks_id] = get_NLG_from_dial(args,task,tasks_id,tokenizer)

    # travis: create a reduced dataset by randomly sampling 2000 train examples and dev/test examples for every domain
    if args.sample_domain != -1 and "TM" in args.dataset_list:  # ['MWOZ_attraction'] 980 86 86
        print(f"sampling {args.sample_domain} train examples and {args.sample_domain//8} dev/test examples")
        for split in datasets.keys():
            size_per_domain = args.sample_domain if split == 'train' else args.sample_domain // 8
            for tasks_id, dataset_task in datasets[split].items(): # aggregate["BYDOMAIN"][split].items()
                dataset_task = [data_task for data_task in dataset_task if len(data_task['history_reply']) < 1700]  # 1400
                datasets[split][tasks_id] = sample(dataset_task, min(len(dataset_task), size_per_domain))

        # len_data_tokenized = defaultdict(list)
        len_data = defaultdict(list)
        for split in datasets.keys():
            for task_id, dataset_task in datasets[split].items():
                for data_task in dataset_task:
                    # input_batch = tokenizer(data_task["history_reply"], padding=True, return_tensors="pt",
                    #                         truncation=False, add_special_tokens=False, return_attention_mask=False)
                    # len_data_tokenized[f"{split}_{task_id}"].append(np.prod(input_batch['input_ids'].size()))
                    len_data[f"{split}_{task_id}"].append(len(data_task['history_reply']))
                len_data[f"{split}_{task_id}"].sort(reverse=True)

    task_id_train = set(task_id for task_id, dataset_task in datasets["train"].items())
    task_id_dev = set(task_id for task_id, dataset_task in datasets["dev"].items())
    task_id_test = set(task_id for task_id, dataset_task in datasets["test"].items())
    common_task_id = list(task_id_train & task_id_dev & task_id_test)
    if (args.multi) and args.fix_dataset:
        common_task_id = common_task_id[:1]

    ### LOGGING SOME INFORMATION ABOUT THE TASKS
    print(f"Tasks: {common_task_id}")
    print(f"Num of Tasks {len(common_task_id)}")
    task = defaultdict(lambda:defaultdict(str))
    for split in ["train","dev","test"]:
        for task_id, dataset_task in datasets[split].items():
            task[task_id][split] = len(dataset_task)
    table = []
    for dom_name, split_len in task.items():
        table.append({"dom":dom_name, "train":split_len["train"], "dev":split_len["dev"], "test":split_len["test"]})
    print(tabulate(table, headers="keys"))

    train_loaders = {}
    valid_loaders = {}
    train_datasets = {}
    val_datasets = {}
    if(args.continual):
        if(not test):
            for task_id, dataset_task in datasets["train"].items():
                if(task_id in common_task_id):
                    train_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.train_batch_size,
                                                        shuffle=True, #pin_memory=True,  # num_workers=3,
                                                        collate_fn=partial(collate_fn_, tokenizer=tokenizer))

                    # train_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
                    train_datasets[task_id] = dataset_task
            for task_id, dataset_task in datasets["dev"].items():
                if(task_id in common_task_id):
                    valid_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size,
                                                        shuffle=False, #pin_memory=True, # num_workers=3,
                                                        collate_fn=partial(collate_fn_, tokenizer=tokenizer))

                    # valid_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))
                    val_datasets[task_id] = dataset_task
    elif(args.multi):
        if(not test):
            dataset_train = []
            for task_id, dataset_task in datasets["train"].items():
                if(task_id in common_task_id):
                    dataset_train += dataset_task
            train_loaders = DataLoader(DatasetTrain(dataset_train), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))

            dataset_dev = []
            for task_id, dataset_task in datasets["dev"].items():
                if(task_id in common_task_id):
                    dataset_dev += dataset_task
            valid_loaders = DataLoader(DatasetTrain(dataset_dev), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))

    # if args.single:
    test_loaders = {}
    test_datasets = {}
    for task_id, dataset_task in datasets["test"].items():
        if (task_id in common_task_id):
            test_loaders[task_id] = DataLoader(DatasetTrain(dataset_task), batch_size=args.valid_batch_size,
                                               shuffle=False,  # pin_memory=True, # num_workers=3,
                                               collate_fn=partial(collate_fn_, tokenizer=tokenizer))
            test_datasets[task_id] = dataset_task
    # else:
        # temp_list = []
        # for task_id, dataset_task in datasets["test"].items():
        #     if(task_id in common_task_id):
        #         temp_list.append(dataset_task)
        # test_datasets = sum(temp_list,[])
        # test_loaders = DataLoader(DatasetTrain(sum(temp_list,[])), batch_size=args.valid_batch_size, shuffle=False,collate_fn=partial(collate_fn_, tokenizer=tokenizer))

    '''
    ### THIS IS JUST FOR CHECKING DUPLICATE DIALOGUES
    testing_dict = defaultdict(list)
    for idx_b, batch in tqdm(enumerate(test_loaders),total=len(test_loaders)):
        for (d_id, t_id, ta_id) in zip(batch["dial_id"],batch["turn_id"],batch["task_id"]):
            if(f'{d_id}_{t_id}_{ta_id}' not in testing_dict):
                testing_dict[f'{d_id}_{t_id}_{ta_id}'].append(1)
            else:
                print(f'{d_id}_{t_id}_{ta_id}')
    '''

    return train_loaders, valid_loaders, test_loaders, (train_datasets,val_datasets,test_datasets), aggregate["DOMAINS_SELECTED"]
