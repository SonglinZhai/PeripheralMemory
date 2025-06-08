# -*- coding=utf-8 -*-
# @Author: Songlin.Zhai
# @Date:   2025-06-01
# @Contact: slinzhai@gmail.com

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import sys
sys.path.append('./')

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Union
import json
import time
import argparse
import copy
import numpy as np
from tqdm import tqdm
from random import sample
from itertools import chain
from transformers import AutoTokenizer


from plms.llama3 import LlamaForCausalLM
from plms.gptj import GPTJForCausalLM
from plms.phi3 import Phi3ForCausalLM
from plms.gemma2 import Gemma2ForCausalLM
from plms.tokenization_gemma import GemmaTokenizer
from kan import MultKAN as KAN  # Official KAN implement is used


CONFIG = {
    "name": "Peripheral Memory for LLMs",
    "gptj":{
        "path": "path/to/GPTJ/",
        "lm_embed_device": "self.model.transformer.wte.weight.device",
        "lm_head_device": "self.model.transformer.ln_f.weight.device"
    },
    "llama3-8B":{
        "path": "path/to/Llama3/8B/",
        "lm_embed_device": "self.model.model.embed_tokens.weight.device",
        "lm_head_device": "self.model.lm_head.weight.device"
    },
    "gemma2-2B":{
        "path": "path/to/Gemma2/2B-it/",
        "lm_embed_device": "self.model.model.embed_tokens.weight.device",
        "lm_head_device": "self.model.lm_head.weight.device"
    },
    "phi3-3.8B":{
        "path": "path/to/Phi3/Phi-3-mini-4k-instruct/",
        "lm_embed_device": "self.model.model.embed_tokens.weight.device",
        "lm_head_device": "self.model.lm_head.weight.device"
    }
}

DATA = {
    "zsre": "./data/zsre_edit.json",
    "counterfact": "./data/counterfact_edit.json"
}

SEGMENTS = [[{'start':   0, 'end':1000}, {'start':2800, 'end':3800}, {'start':4800, 'end':5800}]]


def flatten(data:list): return list(chain.from_iterable(data))

def now(format='%Y-%m-%d-%H:%M:%S'): return time.strftime(format, time.localtime())


class ZsRE(Dataset):
    def __init__(self, data_fp: str) -> None:
        super().__init__()
        with open(data_fp, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
    
    def sample(self, size:int, random=False):
        if random: self._data = sample(self._data, size)
        else: self._data = self._data[:size]

    def segments(self, group_idx, seg_idx=None):
        if seg_idx is None:
            # Combine all segments data in a group, 3K
            data = list()
            for seg in SEGMENTS[group_idx]:
                data.extend(self._data[seg['start']:seg['end']])
            self._data = data
        else:
            # Just Load one setment, 1K
            self._data = self._data[SEGMENTS[group_idx][seg_idx]['start']:\
                                    SEGMENTS[group_idx][seg_idx]['end']]

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return tuple([idx, self._data[idx]])

class CounterFact(Dataset):
    def __init__(self, data_fp: str):
        with open(data_fp, 'r', encoding='utf-8') as f:
            self._data = json.load(f)
    
    def sample(self, size:int, random=False):
        if random: self._data = sample(self._data, size)
        else: self._data = self._data[:size]

    def segments(self, group_idx, seg_idx=None):
        if seg_idx is None:
             # Combine all segments data in a group, 3K
            data = list()
            for seg in SEGMENTS[group_idx]:
                data.extend(self._data[seg['start']:seg['end']])
            self._data = data
        else:
            # Just Load one setment, 1K
            self._data = self._data[SEGMENTS[group_idx][seg_idx]['start']:\
                                    SEGMENTS[group_idx][seg_idx]['end']]
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return tuple([idx, self._data[idx]])

class ZsREDataLoader(DataLoader):
    def __init__(self, dataset:ZsRE, batch_size, shuffle=True):
        super(ZsREDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,
            collate_fn=self.zsre_collate_fn, shuffle=shuffle
        )
    
    def zsre_collate_fn(self, batch):
        batch_idx = [b[0] for b in batch]
        case_ids = [b[1]["case_id"] for b in batch]
        #
        edit_prompts = [b[1]["prompt"] for b in batch]
        edit_ans = [b[1]["target_new"] for b in batch]
        edit_inputs = list(map(lambda x, y: x+' '+y, edit_prompts, edit_ans))
        #
        rep_prompts = [b[1]["paraphrase_prompts"] for b in batch]
        rep_ans = edit_ans
        rep_inputs = list(map(lambda x, y: x+' '+y, rep_prompts, rep_ans))
        #
        loc_prompts = [b[1]["locality_prompt"] for b in batch]
        loc_ans = [b[1]["locality_ground_truth"] for b in batch]
        loc_inputs = list(map(lambda x, y: x+' '+y, loc_prompts, loc_ans))
        #
        return tuple([case_ids,
                      edit_inputs, edit_prompts, edit_ans,
                      rep_inputs, rep_prompts, rep_ans,
                      loc_inputs, loc_prompts, loc_ans])

class CounterFactDataLoader(DataLoader):
    def __init__(self, dataset:CounterFact, batch_size, shuffle=True):
        super(CounterFactDataLoader, self).__init__(
            dataset=dataset, batch_size=batch_size,
            collate_fn=self.cf_collate_fn, shuffle=shuffle
        )
    
    def cf_collate_fn(self, batch):
        # cond = ["{} >> {} || {}".format(b[1]['ground_truth'],
        #                                 b[1]["target_new"],
        #                                 b[1]['prompt']) for b in batch]
        batch_idx = [b[0] for b in batch]
        case_ids = [b[1]["case_id"] for b in batch]
        #
        edit_prompts = [b[1]["prompt"] for b in batch]
        edit_ans = [b[1]["target_new"] for b in batch]
        edit_inputs = list(map(lambda x, y: x+' '+y, edit_prompts, edit_ans))
        #
        rep_prompts = [b[1]["rephrase_prompt"] for b in batch]
        rep_ans = edit_ans
        rep_inputs = list(map(lambda x, y: x+' '+y, rep_prompts, rep_ans))
        #
        loc_prompts = [b[1]["locality_prompt"] for b in batch]
        loc_ans = [b[1]["locality_ground_truth"] for b in batch]
        loc_inputs = list(map(lambda x, y: x+' '+y, loc_prompts, loc_ans))
        #
        return tuple([case_ids,
                      edit_inputs, edit_prompts, edit_ans,
                      rep_inputs, rep_prompts, rep_ans,
                      loc_inputs, loc_prompts, loc_ans])


class Memory(torch.nn.Module):
    def __init__(
        self,
        banks_sizes,
        grid_size=5,
        spline_order=3,
        noise_scale=0.1,
        base_fun='identity', # silu
        grid_eps=0.02, grid_range=[-1, 1],
        sp_trainable=False, sb_trainable=False,
        symbolic_enabled=False,
        save_act=False, auto_save=False, ckpt_path='./model',
        state_id=0, round=0, device='cpu'
    ):
        super(Memory, self).__init__()
        self.bank_sizes = banks_sizes
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_fun = base_fun
        self.memory_width = len(banks_sizes)-1 # The last bank is the confidence bank
        self.memory_depth = banks_sizes[1][1]
        self.device = torch.device('cpu')

        self.memory_banks = torch.nn.ModuleList()
        for bank_size in banks_sizes:
            self.memory_banks.append(
                KAN(width=list(bank_size),
                    grid=grid_size,
                    k=spline_order,
                    noise_scale=noise_scale,
                    base_fun=base_fun,
                    grid_eps=grid_eps, grid_range=grid_range,
                    sp_trainable=sp_trainable, sb_trainable=sb_trainable,
                    symbolic_enabled=symbolic_enabled,
                    save_act=save_act, auto_save=auto_save, ckpt_path=ckpt_path,
                    state_id=state_id, round=round, device=device)
            )

    def forward(self, x:torch.Tensor):
        assert x.dim() < 3, f'[Error in Memory]: Dimension of Memory Input Feature is {x.size()}.'
        assert x.size(-1) == self.memory_width, '[Error in Memory]: Dimension of Memory Input Feature DoesNot Match the Memory Width!'
        if x.dim() == 1: x = torch.unsqueeze(x, dim=0)
        x = x.T
        all_hidden_states = ()
        for x_dim, bank in zip(x, self.memory_banks[:-1]):
            x_dim = torch.unsqueeze(x_dim, dim=-1)
            # Concat the current and past features
            if len(all_hidden_states) != 0: hidden_states = torch.cat((x_dim, all_hidden_states[-1]), dim=-1)
            else: hidden_states = x_dim
            out = bank(hidden_states)
            all_hidden_states = all_hidden_states + (out,)
        # Estimate the attention of the memory feature
        attention = self.memory_banks[-1](all_hidden_states[-1])
        return torch.hstack(all_hidden_states), attention

    def reg_loss(self, reg_metric='edge_forward_spline_n', lamb_l1=1., lamb_entropy=2., lamb_coef=1.0, lamb_coefdiff=1.0, mode='sum'):
        if mode == 'sum':
            return sum(
                bank.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                for bank in self.memory_banks
            )
        else:
            return sum(
                bank.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                for bank in self.memory_banks
            )/(len(self.memory_banks))
    
    def __str__(self) -> str:
        """
        Print:
        model trainable parameters and 
        with number of trainable parameters
        """
        params = 'Memory Parameters Contain:\n'
        for name, param in self.memory_banks.named_parameters():
            if param.requires_grad:
                params = params + '    - ' + name + '     \t->   ' + str(param.size()) + '\n'
        model_parameters = filter(lambda p: p.requires_grad, self.memory_banks.parameters())
        params_count = sum([np.prod(p.size()) for p in model_parameters])
        params_count_M = (params_count/1024)/1024
        return params+'Parameters count: {} -> {} MB\n'.format(params_count,params_count_M)
    
    def to(self, device:torch.device):
        self.memory_banks = self.memory_banks.to(device)
        self.device = device
    
    def save(self, save_name):
        if not os.path.exists('./memory/'): os.makedirs('./memory/')
        save_fp = os.path.join('./memory/', save_name)
        torch.save(self.state_dict(), save_fp)
        print('Save memory to %s'%save_fp)
    
    def load(self, load_name):
        load_fp = os.path.join('./memory/', load_name)
        print('Load memory from %s'%load_fp)
        self.load_state_dict(torch.load(load_fp))


class Convertor(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        tied=False,
        bias=True
    ):
        super(Convertor, self).__init__()
        self.tied = tied
        self.bias = bias
        self.device = torch.device('cpu')
        self.convertor_1 = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=bias)
        self.convertor_2 = torch.nn.Linear(in_features=out_dim, out_features=in_dim, bias=bias)
        if tied:
            del self.convertor_2.weight
            self.convertor_2.weight = self.convertor_1.weight.T
    
    def forward(self, feats, reverse=False):
        if reverse: return self.convertor_2(feats)
        else: return self.convertor_1(feats)
    
    def to(self, device):
        self.convertor_1 = self.convertor_1.to(device)
        if self.tied:
            self.convertor_2.weight.data = self.convertor_2.weight.to(device)
            self.convertor_2.bias.data = self.convertor_2.bias.to(device)
        else: self.convertor_2 = self.convertor_2.to(device)
        self.device = device
    
    def save(self, save_name):
        if not os.path.exists('./convertor/'): os.makedirs('./convertor/')
        save_fp = os.path.join('./convertor/', save_name)
        torch.save(self.state_dict(), save_fp)
        print('Save convertor to %s'%save_fp)
    
    def load(self, load_name):
        load_fp = os.path.join('./convertor/', load_name)
        print('Load convertor from %s'%load_fp)
        self.load_state_dict(torch.load(load_fp))


class PeripheralModel(torch.nn.Module):
    def __init__(
        self,
        llm_name:str,
        memory:Memory
    ):
        super(PeripheralModel, self).__init__()
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.hidden_size = None
        self.num_hidden_layers = None
        self.query_layer_idx = None
        self.merge_layer_idx = None
        self.memory = memory
        self.convertor = None

        name = llm_name.lower()
        if name in ['phi3-3.8b','phi3-7b','phi3-14b']:
            self.llm = 'phi3-3.8B'
            self.model = Phi3ForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG[self.llm]['path'])
            self.replace_idx = 0
        elif name in ['gemma2-2b-it','gemma2-9b-it','gemma2-27b-it']:
            self.llm = 'gemma2-2B'
            self.model = Gemma2ForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = GemmaTokenizer.from_pretrained(CONFIG[self.llm]['path'])
            self.replace_idx = 1
        elif name in ['llama3-8b', 'llama3.1-8b', 'llama3-8b-it', 'llama3.1-8b-it']:
            self.llm = 'llama3-8B'
            self.model = LlamaForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG[self.llm]['path'])
            self.replace_idx = 1
        elif name in ['gpt-j', 'gptj']:
            self.llm = 'gptj'
            self.model = GPTJForCausalLM.from_pretrained(CONFIG[self.llm]['path'])
            self.tokenizer = AutoTokenizer.from_pretrained(CONFIG[self.llm]['path'])
            self.replace_idx = 0
        else: raise Exception(f'The current approach does not support the {name}')

        self.convertor = Convertor(self.model.config.hidden_size, self.memory.memory_width, tied=False)

        self.hidden_size = self.model.config.hidden_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        # Set no grad
        for name, param in self.model.named_parameters(): param.requires_grad = False
        # Setting the pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
    
    def set_query_layer(self, layer_idx): self.query_layer_idx = layer_idx
    def set_merge_layer(self, layer_idx): self.merge_layer_idx = layer_idx

    def _replace_attn_fn(self, attention_masks:torch.Tensor, attention_value, replace_idx:int):
        if type(attention_value) is torch.Tensor:
            assert attention_value.dim() == 1 or attention_value.dim() == 2, 'Error in attention_value dimension!'
        else:
            assert type(attention_value) is int or type(attention_value) is float, 'Error in attention_mask type!'
        assert attention_masks.dim() == 4 or attention_masks.dim() == 2, 'Error in attention_mask dimension!'
        if type(attention_value) is torch.Tensor:
            attention_value = attention_value.to(attention_masks.device)
        if attention_masks.dim() == 4:
            ''' [batch_size, 1, seq_len, seq_len] '''
            if type(attention_value) is torch.Tensor:
                attention_value = attention_value.repeat(1, attention_masks.size(-2))\
                           .view(attention_masks.size(0), 1, attention_masks.size(2))
            attention_masks[:,:,:,replace_idx] = attention_value
        else:
            if type(attention_value) is torch.Tensor:
                attention_value = attention_value if attention_value.dim() == 1 else torch.squeeze(attention_value, -1)
            attention_masks[:,replace_idx] = attention_value
        return attention_masks

    def memory_fn(self, layer_idx, hidden_states, causal_masks, **kwargs):
        if layer_idx == self.query_layer_idx:
            query_feats = hidden_states[torch.arange(hidden_states.size(0)).to(hidden_states.device), self.query_token_idx]
            # query_feats = hidden_states.mean(1)
            self.query_signal = self.convertor(query_feats.to(self.convertor.device))
        if layer_idx == self.merge_layer_idx:
            memory_signal, attn_value = self.memory(self.query_signal.to(self.memory.device)) # attn_value: [batch_size, 1]
            memory_feats = self.convertor(memory_signal.to(self.convertor.device), reverse=True)
            hidden_states[:,self.replace_idx,:] = memory_feats.to(hidden_states.device)
            causal_masks = self._replace_attn_fn(causal_masks, attn_value, self.replace_idx)
        if layer_idx > self.merge_layer_idx:
            causal_masks = self._replace_attn_fn(causal_masks, 1., self.replace_idx)
        return hidden_states, causal_masks
    
    def tokenize(self, batch_inputs:list, batch_prompts:list, add_additional_bos=True):
        if add_additional_bos:
            batch_inputs = list(map(lambda x: self.tokenizer.bos_token+x, batch_inputs))
            batch_prompts = list(map(lambda x: self.tokenizer.bos_token+x, batch_prompts))
        inputs = self.tokenizer(batch_inputs, padding=True, return_tensors='pt')
        seq_length = inputs.attention_mask.sum(1).tolist()
        inputs.attention_mask[:,self.replace_idx] = 0
        labels = inputs.input_ids.detach().clone()
        labels[:,self.replace_idx] = -100
        prompts_ids = self.tokenizer(batch_prompts).input_ids
        prompts_length = list()
        for idx,one_ids in enumerate(prompts_ids):
            # padding_side=Right:  -100... + pre_pred... + pad...
            labels[idx][:len(one_ids)] = -100
            labels[idx][seq_length[idx]:] = -100
            prompts_length.append(len(one_ids))
        return {'input_ids':inputs.input_ids, 'attention_mask':inputs.attention_mask, 'labels':labels}, seq_length, prompts_length
    
    def train(self, batch_inputs, batch_prompts):
        self.optimizer.zero_grad()
        inputs, seq_lengths, prompt_lengths =\
            self.tokenize(batch_inputs, batch_prompts, add_additional_bos=True)
        self.query_token_idx = list(map(lambda x: x-1, prompt_lengths))
        loss = self.model(**inputs, memory_fn=self.memory_fn).loss
        loss.backward()
        self.optimizer.step()
        del self.__dict__['query_signal']
        del self.__dict__['query_token_idx']
        return loss.item()
    
    def test(self, batch_inputs, batch_prompts, add_memory=True, output_hidden_states=True):
        self.model.eval()
        self.convertor.eval()
        self.memory.eval()
        with torch.no_grad():
            if add_memory:
                inputs, seq_lengths, prompt_lengths = self.tokenize(batch_inputs, batch_prompts, add_additional_bos=True)
                del inputs['labels'] # To avoid loss computation for saving time
                self.query_token_idx = list(map(lambda x: x-1, prompt_lengths))
                return self.model(**inputs, memory_fn=self.memory_fn, output_hidden_states=output_hidden_states)
            else:
                inputs, _, _ = self.tokenize(batch_inputs, batch_prompts, add_additional_bos=False)
                del inputs['labels'] # To avoid loss computation for saving time
                return self.model(**inputs, output_hidden_states=output_hidden_states)
    
    def to(self, device:torch.device):
        if self.model is not None:
            self.model = self.model.to(device)
            self.convertor.to(device)
        else: raise Exception('No model specified!')
    
    def __str__(self) -> str:
        """
        Print:
        model trainable parameters and 
        with number of trainable parameters
        """
        params = 'PeripheralModel Trainable Parameters Contain:\n'
        for name, param in self.named_parameters():
            if param.requires_grad:
                params = params + '  - ' + name + '    \t->    ' + str(param.size()) + '\n'
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params_count = sum([np.prod(p.size()) for p in model_parameters])
        params_count_M = (params_count/1024)/1024
        memory_parameters = filter(lambda p: p.requires_grad, self.memory.parameters())
        memory_count = sum([np.prod(p.size()) for p in memory_parameters])
        memory_count_M = (memory_count/1024)/1024
        return params+'Total Parameters count: {} -> {} MB\n'.format(params_count,params_count_M)+\
                      '\nMemory Parameters count: {} -> {} MB\n'.format(memory_count,memory_count_M)


class KME(object):
    def efficacy(self, model, batch_inputs:list, batch_prompts:list, use_memory=True):
        if isinstance(batch_inputs, str): batch_inputs = [batch_inputs,]
        if isinstance(batch_prompts, str): batch_prompts = [batch_prompts,]
        inputs, num_seq, num_prompt = model.tokenize(batch_inputs, batch_prompts, add_additional_bos=use_memory)
        num_answer = list(map(lambda x,y : x-y, num_seq, num_prompt))
        with torch.no_grad():
            outputs = model.test(batch_inputs, batch_prompts, add_memory=use_memory)
        if type(outputs) is torch.Tensor: logits = outputs
        else: logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).tolist()
        answers = list(map(lambda x,i,j: x[i-1:i+j-1], answers,num_prompt,num_answer))
        labels = inputs['labels'].tolist()
        labels = list(map(lambda x,i,j: x[i:i+j], labels,num_prompt,num_answer))
        if -100 in labels: raise Exception('Error in labels when evaluation!')
        res = []
        for ans,label in zip(answers,labels):
            temp_acc = np.mean(np.equal(ans, label))
            if np.isnan(temp_acc): continue
            res.append(temp_acc)
        return res
    
    def locality(self, model, batch_inputs:list, batch_prompts:list, use_memory=False):
        if isinstance(batch_inputs, str): batch_inputs,batch_prompts = [batch_inputs,], [batch_prompts,]
        # Get the model predictions
        inputs, num_seq, num_prompt = model.tokenize(batch_inputs, batch_prompts, add_additional_bos=use_memory)
        num_answer = list(map(lambda x,y : x-y, num_seq, num_prompt))
        with torch.no_grad():
            outputs = model.test(batch_inputs, batch_prompts, add_memory=use_memory)
        if type(outputs) is torch.Tensor: logits = outputs
        else: logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).tolist()
        answers = list(map(lambda x,i,j: x[i-1:i+j-1], answers,num_prompt,num_answer))
        #
        # Obtain the original outputs
        inputs, num_seq, num_prompt = model.tokenize(batch_inputs, batch_prompts, add_additional_bos=False)
        num_answer = list(map(lambda x,y : x-y, num_seq, num_prompt))
        with torch.no_grad():
            outputs = model.test(batch_inputs, batch_prompts, add_memory=False)
        if type(outputs) is torch.Tensor: logits = outputs
        else: logits = outputs.logits
        labels = torch.argmax(logits, dim=-1).tolist()
        labels = list(map(lambda x,i,j: x[i-1:i+j-1], labels,num_prompt,num_answer))
        res = []
        for ans,label in zip(answers,labels):
            temp_acc = np.mean(np.equal(ans, label))
            if np.isnan(temp_acc): continue
            res.append(temp_acc)
        return res

    def __call__(self, running_config:dict, model:PeripheralModel, save_res=False):
        self.config = running_config
        print(f'Model edit settings:\n', self.config, '\n', '--'*15)
        if self.config['data_name'] == 'zsre':
            data = ZsRE(DATA['zsre'])
            data.segments(self.config['group_idx'], self.config['seg_idx'])
            edit_loader = ZsREDataLoader(data, batch_size=self.config['train_bs'], shuffle=True)
            test_loader = ZsREDataLoader(data, batch_size=self.config['test_bs'], shuffle=False)
        elif self.config['data_name'] == 'counterfact':
            data = CounterFact(DATA['counterfact'])
            data.segments(self.config['group_idx'], self.config['seg_idx'])
            edit_loader = CounterFactDataLoader(data, batch_size=self.config['train_bs'], shuffle=False)
            test_loader = CounterFactDataLoader(data, batch_size=self.config['test_bs'], shuffle=False)
        #
        # Pre-Evaluation
        acc_log = dict()
        loc_log = dict()
        rep_log = dict()
        print('\nModel evaluation before the model editing.')
        efficacy_acc = list()
        rephrase_acc = list()
        locality_acc = list()
        pbar = tqdm(total=data.__len__(), ncols=75, leave=True)
        pbar.set_description_str(desc='Pre->')
        for _, batch in enumerate(test_loader):
            efficacy_acc.append(self.efficacy(model, batch[1], batch[2], use_memory=False))
            rephrase_acc.append(self.efficacy(model, batch[4], batch[5], use_memory=False))
            locality_acc.append(self.locality(model, batch[7], batch[8], use_memory=False))
            pbar.update(len(batch[0]))
        pbar.refresh()
        print(f'\nMean Efficacy: {np.mean(flatten(efficacy_acc))}',
              f'\nMean Rephrase: {np.mean(flatten(rephrase_acc))}',
              f'\nMean Locality: {np.mean(flatten(locality_acc))}')
        acc_log['pre'] = round(np.mean(flatten(efficacy_acc)), 6)
        loc_log['pre'] = round(np.mean(flatten(locality_acc)), 6)
        rep_log['pre'] = round(np.mean(flatten(rephrase_acc)), 6)
        #
        # Model editing
        print('\nStart model editing...')
        acc_log['post_best'], acc_log['post_worst'] = None, None
        rep_log['post_best'], rep_log['post_worst'] = None, None
        loc_log['post_best'], loc_log['post_worst'] = None, None
        acc_log['post_epoch'] = list()
        rep_log['post_epoch'] = list()
        loc_log['post_epoch'] = list()
        efficacy_acc_test = list()
        rephrase_acc_test = list()
        locality_acc_test = list()
        rephrase_acc_best = 0.
        epoch_loss = list()
        epoch_size = self.config['epoch']
        start_time = now()
        for epoch in range(epoch_size):
            print(f'\nEpoch {epoch+1}:')
            losses = list()
            pbar.set_description_str(desc='Edit->')
            pbar.reset()
            for _, batch in enumerate(edit_loader):
                loss = model.train(batch[1], batch[2])
                pbar.update(len(batch[0]))
                losses.append(loss)
            pbar.refresh()
            print(f'\n[Mean Iter Loss ({epoch+1}/{epoch_size})]: {np.mean(losses)}')
            epoch_loss.append(np.mean(losses))
            #
            # Post-Evaluation
            torch.cuda.empty_cache()
            efficacy_acc = list()
            rephrase_acc = list()
            locality_acc = list()
            pbar.set_description_str(desc='Post->')
            pbar.reset()
            for idx, batch in enumerate(test_loader):
                efficacy_acc.append(self.efficacy(model, batch[1], batch[2], use_memory=True))
                rephrase_acc.append(self.efficacy(model, batch[4], batch[5], use_memory=True))
                locality_acc.append(self.locality(model, batch[7], batch[8], use_memory=False))
                pbar.update(len(batch[0]))
            pbar.refresh()
            print(f'\n[Efficacy ({epoch+1}/{epoch_size})]: {np.mean(flatten(efficacy_acc))}')
            print(f'[Rephrase ({epoch+1}/{epoch_size})]: {np.mean(flatten(rephrase_acc))}')
            print(f'[Locality ({epoch+1}/{epoch_size})]: {np.mean(flatten(locality_acc))}')
            acc_log['post_epoch'].append(np.mean(flatten(efficacy_acc)))
            rep_log['post_epoch'].append(np.mean(flatten(rephrase_acc)))
            loc_log['post_epoch'].append(np.mean(flatten(locality_acc)))
            end_time = now()
            print(f'[Epoch {epoch+1} Start/Finish  Time]: {start_time} / {end_time}')
            if np.mean(flatten(rephrase_acc)) > rephrase_acc_best:
                rephrase_acc_best = np.mean(flatten(rephrase_acc))
                efficacy_acc_test = flatten(efficacy_acc)
                rephrase_acc_test = flatten(rephrase_acc)
                locality_acc_test = flatten(locality_acc)
        pbar.close()
        
        acc_log['post_best'], acc_log['post_worst'] = round(max(acc_log['post_epoch']), 6), round(min(acc_log['post_epoch']), 6)
        rep_log['post_best'], rep_log['post_worst'] = round(max(rep_log['post_epoch']), 6), round(min(rep_log['post_epoch']), 6)
        loc_log['post_best'], loc_log['post_worst'] = round(max(loc_log['post_epoch']), 6), round(min(loc_log['post_epoch']), 6)
        print(f'Efficacy {acc_log["pre"]} -> best: {round(max(acc_log["post_epoch"]), 6)}, worst: {round(min(acc_log["post_epoch"]), 6)}')
        print(f'Rephrase {rep_log["pre"]} -> best: {round(max(rep_log["post_epoch"]), 6)}, worst: {round(min(rep_log["post_epoch"]), 6)}')
        print(f'Locality {loc_log["pre"]} -> best: {round(max(loc_log["post_epoch"]), 6)}, worst: {round(min(loc_log["post_epoch"]), 6)}')
        print(f'Model edit settings:\n', self.config, '\n', '--'*15+'\n')
        res = {"Efficacy":acc_log, "Generality":rep_log, "Locality":loc_log, "Epoch_Loss":epoch_loss, "Efficacy_CaseByCase_Record":efficacy_acc_test, "Generality_CaseByCase_Record":rephrase_acc_test, "Locality_CaseByCase_Record":locality_acc_test}
        if save_res:
            llm_name = model.llm.replace('-','.')
            with open(llm_name+'.'+str(model.query_layer_idx)+'_'+str(self.config['data_name'])+'_'+str(model.memory.memory_width)+'.'+str(model.memory.memory_depth)+'.'+str(model.memory.grid_size)+'_'+str(self.config['group_idx'])+'.'+str(self.config['seg_idx'])+'.json', 'w', encoding='utf-8') as ofstream:
                json.dump(res, ofstream, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_name', type=str, required=True)
    parser.add_argument('--data_name', type=str, required=True, choices=['zsre', 'counterfact'])
    parser.add_argument('--group_idx', type=int, default=0)
    parser.add_argument('--seg_idx', type=int, default=None)
    parser.add_argument('--memory_width', type=int, default=512)
    parser.add_argument('--memory_depth', type=int, default=11)
    parser.add_argument('--grid_size', type=int, default=9)
    parser.add_argument('--query_layer', type=int, default=23, help='Query Layer should be in range [0, L-1].')
    parser.add_argument('--epoch', type=int, default=50, help='Memory writing epoch.')
    parser.add_argument('--train_bs', type=int, default=10, help='Train Batch size.')
    parser.add_argument('--test_bs', type=int, default=10, help='Test Batch size.')
    parser.add_argument('--save_res', type=bool, default=False, help='Save the evaluation results.')
    args = parser.parse_args()
    #
    running_config = {
        'llm_name': args.llm_name, 'query_layer':args.query_layer,
        'data_name': args.data_name, 'group_idx': args.group_idx, 'seg_idx': args.seg_idx,
        'memory_width': args.memory_width, 'memory_depth': args.memory_depth, 'grid_size':args.grid_size,
        'epoch':args.epoch, 'train_bs': args.train_bs, 'test_bs': args.test_bs
    }
    #
    memory = Memory([(1,int(np.ceil(args.memory_depth/2)),1)]+\
                    [(2,args.memory_depth,1)]*(args.memory_width-1)+\
                    [(1,int(np.ceil(args.memory_depth/2)),1)],
                    grid_size=args.grid_size)
    memory.to('cuda:0')
    #
    model = PeripheralModel(running_config['llm_name'], memory)
    model.set_query_layer(args.query_layer)
    model.set_merge_layer(args.query_layer)
    model.to('cuda:0')
    print(model.model)
    print(model)
    #
    task = KME()
    task(running_config, model, save_res=args.save_res)
