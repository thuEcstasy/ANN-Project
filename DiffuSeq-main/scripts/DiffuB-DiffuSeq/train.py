#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append('.')
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["WANDB_MODE"] = "offline"


# In[2]:


import argparse
import json
import math
import random
import numpy as np
import socket
import blobfile as bf
import io
import glob
import time
from tqdm import tqdm
import copy
import functools
import gc
from functools import partial

from diffuseq.utils import logger
from diffuseq.utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
    mean_flat,
    update_ema
)
from diffuseq.utils.fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)

from transformers import set_seed, AutoTokenizer, PreTrainedTokenizerFast, AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

import wandb

import torch as th
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.optim import AdamW
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

import psutil
import datasets
from datasets import Dataset as Dataset2

from abc import ABC, abstractmethod

import csv
import subprocess as sp


# In[3]:


GPUNumber = 0

seed = 102

learning_rate = 1e-4
hidden_t_dim = 128
hidden_dim = 128
dropout = 0.1
weight_decay = 0.0
  
diffusion_steps = 2000
ema_rate = 0.9999
schedule_sampler_name = 'lossaware'
noise_schedule = 'sqrt'
timestep_respacing = ''

dataset = 'QQP'
data_dir = f'/home/aiuser/ANN/DiffuSeq-main/datasets/QQP'
vocab = 'bert'
seq_len = 128

use_plm_init = 'no'

config_name = '/home/aiuser/ANN/DiffuSeq-main/models/bert-base-uncased'

diffusion_ver = 'metaBeta_ver'
checkpoint_path = f'checkpoint/{dataset}/{diffusion_ver}'
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

use_fp16 = False
gradient_clipping = -1.0 
learn_sigma = False
use_kl = False
predict_xstart = True
rescale_timesteps = True
rescale_learned_sigmas = False
sigma_small = False
emb_scale_factor = 1.0


# ## Funciton

# ### dist_util

# In[4]:


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(GPUNumber)
    
    dist.init_process_group(backend=backend, init_method="env://")
    
def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()


# ### basic_utils

# In[5]:


class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, config_name, checkpoint_path, vocab):
        if vocab == 'bert':
            config_name = "/home/aiuser/ANN/DiffuSeq-main/models/bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            tokenizer.save_pretrained(checkpoint_path)
        else: 
            # load vocab from the path
            print('#'*30, 'load vocab from', vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]':2, '[PAD]':3}
            with open(vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0:
                path_save_vocab = f'{checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                
        self.vocab_size = len(self.tokenizer)

    def get_vocab_size(self):
        return self.vocab_size
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            # special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens
    
def load_tokenizer(config_name, checkpoint_path, vocab):
    tokenizer = myTokenizer(config_name, checkpoint_path, vocab)
    vocab_size = tokenizer.get_vocab_size()
    return tokenizer, vocab_size

def load_model_emb(vocab_size, hidden_dim, checkpoint_path):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = th.nn.Embedding(vocab_size, hidden_dim)
    path_save = '{}/random_emb.torch'.format(checkpoint_path)
    path_save_ind = path_save + ".done"
    if int(os.environ['LOCAL_RANK']) == GPUNumber:
        if os.path.exists(path_save):
            # print('reload the random embeddings', model)
            model.load_state_dict(th.load(path_save))
        else:
            print('initializing the random embeddings', model)
            # random Gaussian embeddings as well as pre-trained word embedding (from diffusion-LM paper)
            th.nn.init.normal_(model.weight)
            th.save(model.state_dict(), path_save)
            os.sync() # It is used to force write of everything to disk.
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        # print('reload the random embeddings', model)
        model.load_state_dict(th.load(path_save))

    return model

def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    betas,
    # noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


# ### load data

# In[6]:


def load_data_text(
    batch_size, 
    seq_len, 
    dataset, 
    data_dir,
    deterministic=False,  
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(dataset, data_dir, seq_len, split=split, loaded_vocab=loaded_vocab)
    
    dataset = TextDataset(
        training_data,
        model_emb=model_emb
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        shuffle=not deterministic,
        num_workers=0,
    )
    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y}

        return result_dict

    # Tokenize the data x and y
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)
    print('### tokenized_datasets...example', tokenized_datasets['input_id_x'][0])
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)
            
            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    # Merge data x+[sep]+y
    # Mask  data 0(x+[sep])
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # trim data and add padding
    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(dataset, data_dir, seq_len, split='train', loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(dataset, data_dir))

    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        print('### Loading form the TRAIN set...')
        path = f'{data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading form the VALID set...')
        path = f'{data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading form the TEST set...')
        path = f'{data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for row in f_reader:
            sentence_lst['src'].append(json.loads(row)['src'].strip())
            sentence_lst['trg'].append(json.loads(row)['trg'].strip())

    print('### Data samples...\n', sentence_lst['src'][:2], sentence_lst['trg'][:2])
        
    # get tokenizer.
    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with th.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(th.tensor(input_ids).to(dev()))
            
            arr = np.array(hidden_state.cpu(), dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])

            return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = th.full([len(examples), max_length], pad_token_id, dtype=th.int64).tolist()
    mask_ = th.full([len(examples), max_length], pad_token_id, dtype=th.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


# ### gaussian_diffusion

# In[7]:


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param predict_xstart: the model outputs to predict x_0, else to predict eps.
    :param learn_sigmas: the model outputs to predict sigma or not. Default: False
    :param rescale_learned_sigmas, sigma_small: details setting of learned sigmas
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        rescale_timesteps=False,
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[1])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=1)
        self.alphas_cumprod_prev = np.insert(self.alphas_cumprod, 0, 1.0, axis=1)[:, :-1]
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.insert(self.posterior_variance, 1, self.posterior_variance[:, 1], axis=1)[:, 1:]
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.mapping_func = None # implement in train main()
        self.add_mask_noise = False # TODO
        

    def training_losses(self, model, *args, **kwargs):
        self.model = model
        return self.training_losses_seq2seq(model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, offsets, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape, offsets) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, offsets) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart, offsets):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape, offsets) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, offsets)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t, offsets):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape, offsets) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape, offsets)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape, offsets
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, offsets, noise=None, mask=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param mask: anchoring masked position
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape
        
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape, offsets) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, offsets)
            * noise
        )

        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
            return th.where(mask==0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t, offsets):
        """
        Compute the mean and variance of the diffusion posterior: 
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape, offsets) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape, offsets) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape, offsets)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape, offsets
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, offsets, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.size(0), x.size(-1)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        
        # for fixedlarge, we set the initial (log-)variance like so
        # to get a better decoder log likelihood.
        model_variance = np.insert(self.betas, 1, self.posterior_variance[:, 1], axis=1)[:, 1:]
        model_log_variance = np.log(model_variance)
        
        model_variance = _extract_into_tensor(model_variance, t, x.shape, offsets)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape, offsets)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            # knn rounding -> using most close token as input, and get new emb output
            pred_xstart = process_xstart(model_output)
        else:
            ### model is used to predict eps
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, offsets=offsets, eps=model_output)
            )

        # μ_t
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t, offsets=offsets
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self, model, x, t, offsets, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            offsets,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if top_p is not None and top_p > 0:
            # print('top_p sampling')
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()

        else:
            noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # get new sample from mean and variance 
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        if mask == None:
            pass
        else:
            # only denoise y, x keep use x0
            sample = th.where(mask==0, x_start, sample)

        return {
            "sample": sample, 
            "pred_xstart": out["pred_xstart"], # after knn rounding model output
            "greedy_mean": out["mean"], 
            "out": out
        }

    
    def p_sample_loop(
        self,
        model,
        shape,
        offsets,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param mask: anchoring masked position to x_start
        :param clamp_step: in clamp_first mode, choose end clamp step, otherwise starting clamp step
        :param clamp_first: bool, clamp_first mode
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = []
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            offsets,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start
        ):
            final.append(sample['sample'].tolist())
        return final

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        offsets,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None: # custom your the start point of x_0
            sample_x = noise # mask x
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices: # from T to 0
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = denoised_fn
                else:
                    denoised_fn_cur = None
            with th.no_grad():
                out = self.p_sample(
                    model,
                    sample_x,
                    t,
                    offsets,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    top_p=top_p,
                    mask=mask,
                    x_start=x_start
                )
                if i == 0:
                    yield out
                sample_x = out["sample"]


    def _get_x_start(self, x_start_mean, std):
        '''
        Word embedding projection from {Emb(w)} to {x_0}
        :param x_start_mean: word embedding
        :return: x_0
        '''
        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
             x_start_mean + std * noise
        )

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):
        '''
        the loss of -log p(w|z_0)
        :param x_start_mean: word embedding
        :return: x_0
        '''
        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask != None:
            decoder_nll *= mask # only y
        if mask != None:
            decoder_nll = decoder_nll.sum(dim=-1)/mask.sum(dim=-1) # each y mean nll loss
        else:
            decoder_nll = decoder_nll.mean(dim=-1) # each sentence mean nll loss

        return decoder_nll

    def _x0_helper(self, model_output, x, t, offsets):

        if self.predict_xstart:
            pred_xstart = model_output
            # pred_prev -> q(x_{t-1} | x_t, x_0) mean
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t, offsets=offsets
            )

        else: # predict eps
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, offsets=offsets, eps=model_output)
        
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t, offsets=offsets
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}

    def training_losses_seq2seq(self, model, x_start, t, offsets, ids, mask, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. # not used unless fixing the input embeddings
        :param t: a batch of timestep indices.
        :param ids: from origial model_kwargs dict
        :param mask: from origial model_kwargs dict
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        x_start_fix = x_start
        input_ids_x = ids
        input_ids_mask = mask
        
        x_start_mean = model.model.module.get_embeds(input_ids_x)
        
        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]*input_ids_x.shape[0]).to(x_start_mean.device),
                                   x_start_mean.shape,
                                   offsets)

        x_start = self._get_x_start(x_start_mean, std)
        if noise is None:
            noise = th.randn_like(x_start)

        x_t = self.q_sample(x_start, t, offsets, noise=noise, mask=input_ids_mask)
        
        get_logits = model.model.module.get_logits

        terms = {}

        target = x_start
        model_output = model(x_t, self._scale_timesteps(t))
        assert model_output.shape == target.shape == x_start.shape
        terms["mse"] = mean_flat((target - model_output) ** 2)
        
        model_out_x_start = self._x0_helper(model_output, x_t, t, offsets)['pred_xstart'] # predicted_xstart = model_output
        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_out_x_start) ** 2)
        terms["mse"] = th.where(t0_mask, t0_loss, terms["mse"])

        out_mean, _, _ = self.q_mean_variance(x_start, th.LongTensor([self.num_timesteps - 1]*input_ids_x.shape[0]).to(x_start.device), offsets)
        tT_loss =  mean_flat(out_mean ** 2)
        
        decoder_nll = self._token_discrete_loss(x_start, get_logits, input_ids_x) # embedding regularization
        terms["nll"] = self._token_discrete_loss(model_out_x_start, get_logits, input_ids_x, mask=input_ids_mask, truncate=True, t=t) # x_0->model_out_x_start
        
        terms["loss"] = terms["mse"] + decoder_nll + tT_loss

        return terms

def _extract_into_tensor(arr, timesteps, broadcast_shape, offsets):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    start_index = offsets*timesteps.shape[0]
    res = th.tensor([arr[start_index+i][timesteps[i]] for i in range(timesteps.shape[0])]).float().to(device=timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = kwargs["betas"].shape[1]

        base_diffusion = GaussianDiffusion(**kwargs) 
        last_alpha_cumprod = 1.0
        new_betas = []
        for i in range(kwargs['betas'].shape[1]):
            if i in self.use_timesteps:
                new_betas.append(1 - base_diffusion.alphas_cumprod[:, i] / last_alpha_cumprod)
                last_alpha_cumprod = base_diffusion.alphas_cumprod[:, i]
                self.timestep_map.append(i)
        kwargs["betas"] = np.transpose(np.array(new_betas))
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called p_mean_var')
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        # print('called training_losses')
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
    
    def get_sqrt_alphas_cumprod(self):
        return super().get_sqrt_alphas_cumprod()


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


# ### transformer model

# In[8]:


class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size

        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )
        
        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", th.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", th.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1))
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))

        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x

        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h


# ### step_sample

# In[9]:


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "lossaware":
        return LossSecondMomentResampler(diffusion)
    elif name == "fixstep":
        return FixSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch. #(from diffuseq p.4)

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion_steps):
        self.diffusion = diffusion_steps
        self._weights = np.ones([diffusion_steps])

    def weights(self):
        return self._weights

class FixSampler(ScheduleSampler):
    def __init__(self, diffusion_steps):
        self.diffusion_steps = diffusion_steps

        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        self._weights = np.concatenate([np.ones([diffusion_steps//2]), np.zeros([diffusion_steps//2]) + 0.5])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)
        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)
        
        del batch_sizes, max_bs, timestep_batches, loss_batches, timesteps, losses

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion_steps, history_per_term=10, uniform_prob=0.001):
        self.diffusion_steps = diffusion_steps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion_steps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion_steps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


# ### Model B  (Encoder-Decoder)

# In[12]:


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True) 
        
    def forward(self, input_sentences):
        self.lstm.flatten_parameters()
        embedded = self.embedding(input_sentences)
        encoder_output, (encoder_h, encoder_c) = self.lstm(embedded)
        return encoder_output, encoder_h, encoder_c
    
class Decoder(nn.Module):
    def __init__(self, output_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_size+1, embedding_dim) # output_size+1 -> T/F + BOS
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_instructions, encoder_h, encoder_c):
        self.lstm.flatten_parameters()
        embedded = self.embedding(input_instructions)
        decoder_output, (decoder_h, decoder_c) = self.lstm(embedded, (encoder_h, encoder_c))
        output = self.linear(decoder_output)
        output = self.softmax(output)
        return output


# ### other functions 

# In[13]:


def create_log_file(filename):
    with open(f'{checkpoint_path}/{filename}.csv', 'w' , newline='') as csvfile:
        writer = csv.writer(csvfile)

def write_log(msg):
    with open(f'{checkpoint_path}/training_log.csv', 'a' , newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([msg])
        
def write_time_usage(msg):
    with open(f'{checkpoint_path}/time_usage.csv', 'a' , newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([msg])
        
def record_information(checkpoint_path, msg, filename):
    with open(f'{checkpoint_path}/{filename}.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([msg])
        
def sample_sentence(data, mask, num, N, seed = None):
    idx_list = []
    if seed != None:
        random.seed(seed)
    while(len(idx_list) < ((num-1) // N + 1)):
        idx = random.randint(0, len(data)-1)
        if idx not in idx_list: idx_list.append(idx)
    batch_data = th.tensor([data[idx] for idx in idx_list for _ in range(N)])[:num]
    batch_mask = th.tensor([mask[idx] for idx in idx_list for _ in range(N)])[:num]
    return batch_data, batch_mask, idx_list

def get_standard_betas(diffusion_steps, noise_schedule):
    if noise_schedule == 'linear':
        scale = 1000 / diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, diffusion_steps, dtype=np.float64)
    elif noise_schedule == 'sqrt':
        return betas_for_alpha_bar(diffusion_steps, lambda t: 1-np.sqrt(t + 0.0001),)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def calculate_my_betas(beta_ins, standard_betas):
    betas = list()
    for i in range(len(beta_ins)):
        point = 0
        now = 0
        standard_betas_copy = standard_betas.copy()
        beta_temp = [standard_betas_copy[point]]
        point += 1
        now = point
        for instruction in beta_ins[i]:
            if instruction: 
                beta_temp.append(standard_betas_copy[point])
                now = point
            else: 
                beta_temp.append(standard_betas_copy[now])
            point += 1
        betas.append(beta_temp)
    betas = th.tensor(betas)
    return betas

def _scale_timesteps(t, rescale_timesteps, num_timesteps):
    if rescale_timesteps:
        return t.float() * (1000.0 / num_timesteps)
    return t

def modelB_inference(batch_data, encoder, decoder, diffusion_steps, word2idx, mode, weight):
    s_t = time.time()
    encoder.eval()
    decoder.eval()
    
    TF_weight = th.tensor([(1+weight), (1-weight)]).to(dev())
    
    with th.no_grad():
        encoder_output, encoder_h, encoder_c = encoder(batch_data.to(dev()))

        decoder_input = th.tensor([[word2idx['BOS']]*(diffusion_steps-1) for _ in range(batch_data.shape[0])])
        decoder_input = decoder_input.to(dev())

        inference_data = th.tensor([[word2idx['F']]*(diffusion_steps-1) for _ in range(decoder_input.shape[0])])
        for step in range(diffusion_steps-1):
            decoder_output = decoder(decoder_input, encoder_h, encoder_c)
            decoder_output = decoder_output[:,step,]
            decoder_output = decoder_output * TF_weight
            if mode == 'argmax':
                decoder_output_TF = th.argmax(decoder_output, dim=1)
            elif mode == 'multinomial':
                decoder_output_TF = th.multinomial(decoder_output, 1)
                decoder_output_TF = decoder_output_TF[:,0]

            inference_data[:, step] = decoder_output_TF
            if step+1 < diffusion_steps-1:
                decoder_input[:,step+1] = decoder_output_TF
            del decoder_output, decoder_output_TF
        del encoder_output, encoder_h, encoder_c, decoder_input
    e_t = time.time() - s_t
    write_time_usage(f'[Model B Inference] batch:{batch_data.shape[0]}. Using {e_t/60:.2f} mins.')
    return inference_data

def get_loss_reward(
    betas, 
    data, 
    mask, 
    model, 
    diffusion_steps,
    t,
    offsets,
    timestep_respacing, 
    rescale_timesteps,
    predict_xstart,
    learn_sigma,
    use_kl,
    rescale_learned_sigmas,
):
    s_t = time.time()
    model.eval()
    with th.no_grad():
        # create diffusion with betas
        if not timestep_respacing:
            timestep_respacing = [diffusion_steps]

        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=rescale_timesteps,
            predict_xstart=predict_xstart,
            learn_sigmas = learn_sigma,
            sigma_small = sigma_small,
            use_kl = use_kl,
            rescale_learned_sigmas=rescale_learned_sigmas
        )

        # get q sample with each diffusion_steps
        x_start_mean = model.module.get_embeds(data) # shape -> (batch_size, seq_len, hidden_dim)
        std = _extract_into_tensor(
            diffusion.sqrt_one_minus_alphas_cumprod,
            th.tensor([0]*x_start_mean.shape[0]).to(x_start_mean.device),
            x_start_mean.shape,
            offsets,
        ).to(dev())

        x_start_q = diffusion._get_x_start(x_start_mean, std) # shape -> (batch_size, seq_len, hidden_dim)
        q_mask = mask

        noise = th.randn_like(x_start_q)

        
        # calculate each diffusion_steps loss
        target = x_start_q
        
        reward_loss = []
        q = diffusion.q_sample(x_start_q, t, offsets, noise=noise, mask=q_mask).to(dev())
        model_output = model(q, _scale_timesteps(t, rescale_timesteps, diffusion_steps))

        mse = mean_flat((target - model_output) ** 2)

        t0_mask = (t == 0)
        t0_loss = mean_flat((x_start_mean - model_output) ** 2)
        mse = th.where(t0_mask, t0_loss, mse)
        
        mse = 1.0 / mse

        reward_loss.append(mse.tolist())

        reward_loss = th.tensor(reward_loss)
        # shape: (diffusion_steps, batch) -> (batch, diffusion_steps)
        reward_loss = th.transpose(reward_loss, 0, 1)
    e_t = time.time() - s_t
    write_time_usage(f'[Get loss reward] batch:{data.shape[0]}. Using {e_t/60:.2f} mins.')
    return reward_loss

def save_model(model, model_name, master_params, step = None):
    if step == None:
        filename = f'model/{model_name}.pt'
    else:
        filename = f'model/{model_name}_{step}.pt'
    state_dict = model.state_dict()
    for i, (name, _value) in enumerate(model.named_parameters()):
        assert name in state_dict
        state_dict[name] = master_params[i]
    msg = 'writing to', bf.join(checkpoint_path, filename)
    write_log(msg)
    with bf.BlobFile(bf.join(checkpoint_path, filename), "wb") as f: # DEBUG **
        th.save(state_dict, f)
        
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=index,memory.used --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    return memory_free_info


# # Main

# ## tokenizer and model_emb

# In[14]:

setup_dist() # Setup a distributed process group. 
logger.configure()

print(f'seed: {seed}')
set_seed(seed)

tokenizer, vocab_size = load_tokenizer(config_name, checkpoint_path, vocab)
model_weight = load_model_emb(vocab_size, hidden_dim, checkpoint_path)
model_weight.to(dev())

# ## initial hyperparameters

# In[15]:


batch_size = 2048
micro_batch_size_B = 1024
micro_batch_size_D = 64

update_D_loop = 100
update_D_count = 0

weight = 0 # Rsparse choose T/F distribution * weight => 希望能多選一些F

exploration = 1
epochs = 80000
save_interval = 1000

name2val = float(0)
name2cnt = int(0)


# ## load data

# In[16]:


data = load_data_text(
batch_size=batch_size,
seq_len=seq_len,
dataset = dataset,
data_dir = data_dir,
loaded_vocab=tokenizer,
model_emb=model_weight # use model's weights as init
)


# ## initial model $B_\phi$ (encoder-decoder), model $f_\theta$

# In[17]:


word2idx = {
    'F': 0,
    'T': 1,
    'BOS': 2,
} 


# In[18]:


input_size = vocab_size
output_size = 2 # T/F
encoder_embedding_dim = 64
decoder_embedding_dim = 64
encoder_hidden_dim = 64
decoder_hidden_dim = 64

encoder = Encoder(input_size, encoder_embedding_dim, encoder_hidden_dim).to(dev())
decoder = Decoder(output_size, decoder_embedding_dim, decoder_hidden_dim).to(dev())

encoder_params = copy.deepcopy(list(encoder.parameters()))
decoder_params = copy.deepcopy(list(decoder.parameters()))

encoder_optimizer = optim.Adam(encoder_params)
decoder_optimizer = optim.Adam(decoder_params)
criterion = nn.NLLLoss(reduction = 'none')

# DP
encoder_DP = th.nn.DataParallel(encoder)
decoder_DP = th.nn.DataParallel(decoder)


# In[19]:


# beta schedule
standard_betas = get_standard_betas(diffusion_steps, noise_schedule)


# In[20]:


modelD = TransformerNetModel(
    input_dims=hidden_dim,
    output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
    hidden_t_dim=hidden_t_dim,
    dropout=dropout,
    config_name=config_name,
    vocab_size=vocab_size,
    init_pretrained=use_plm_init
)
modelD.to(dev())


schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion_steps)

master_params = list(modelD.parameters())
ema_params = copy.deepcopy(master_params)
optimizer = AdamW(master_params, lr=learning_rate, weight_decay=weight_decay)

# DP
modelD_DP = th.nn.DataParallel(modelD)


# In[21]:


# initial log file
log_file_name = [
    'time_usage',
    'training_log',
    'meta_rewards',
    'train_x_beta_ins',
]
for filename in log_file_name: create_log_file(filename)


# ## training loop

# In[22]:


for epoch in range(epochs):
    start_time = time.time()
    print(f'Epoch {epoch+1} / {epochs}')
    write_log(f'Epoch {epoch+1} / {epochs}')

    batch_hidden, batch = next(data)
    batch_hidden = batch_hidden.to(dev())
    batch_data = batch['input_ids']
    batch_mask = batch['input_mask']
    batch_data_x = batch_data * (1 - batch_mask) # only x mask y

    # R base
    if epoch % update_D_loop == 0:
        beta_ins = modelB_inference(batch_data_x, encoder_DP, decoder_DP, diffusion_steps, word2idx, 'multinomial', weight)
        betas = calculate_my_betas(beta_ins, standard_betas)            
        t, weights = schedule_sampler.sample(batch_data.shape[0], dev())

        reward_base = get_loss_reward(
            betas, 
            batch_data.to(dev()), 
            batch_mask.to(dev()),
            modelD_DP,
            diffusion_steps,
            t,
            0, # offsets
            timestep_respacing, 
            rescale_timesteps,
            predict_xstart,
            learn_sigma,
            use_kl,
            rescale_learned_sigmas,
        )

    # update model D
    if epoch % update_D_loop != 99:
        beta_ins = modelB_inference(batch_data_x, encoder_DP, decoder_DP, diffusion_steps, word2idx, 'multinomial', weight)
        betas = calculate_my_betas(beta_ins, standard_betas)

        # diffusion
        if not timestep_respacing:
            timestep_respacing = [diffusion_steps]

        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=rescale_timesteps,
            predict_xstart=predict_xstart,
            learn_sigmas = learn_sigma,
            sigma_small = sigma_small,
            use_kl = use_kl,
            rescale_learned_sigmas=rescale_learned_sigmas
        )

        # calculate lossD
        modelD_DP.train()
        zero_grad(master_params)

        loop = ((batch_data.shape[0]-1) // micro_batch_size_D)+1
        for i in range(loop):
            t, weights = schedule_sampler.sample(batch_hidden[i*micro_batch_size_D:(i+1)*micro_batch_size_D].shape[0], dev())

            compute_losses = functools.partial(
                diffusion.training_losses,
                modelD_DP,
                batch_hidden[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
                t.to(dev()),
                i, # offset
                batch_data[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
                batch_mask[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
            )

            losses = compute_losses()

            if isinstance(schedule_sampler, LossAwareSampler):
                schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = losses['loss'] * weights
            loss= loss.mean()


            # output loss calculate
            if not th.isnan(loss):
                name2val = name2val * name2cnt / (name2cnt + 1) + loss.item() / (name2cnt + 1)
                name2cnt = name2cnt + 1
            else:
                write_log(f'[Warning] loss nan')

            loss.backward()
            del losses, loss, t, weights
        # _anneal_lr
        frac_done = (epoch + 1) / epochs
        lr = learning_rate * (1 - frac_done)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # update model D
        optimizer.step()
        # update ema_params
        update_ema(ema_params, master_params, rate=ema_rate)
        write_log(f'[Model D] loss: {name2val:.4f}')

    else: # Meta
        betas_ins_list = list()
        betas_list = list()
        for e in range(exploration):
            write_log(f'[Exploration 1] {e+1} / {exploration}')
            beta_ins = modelB_inference(batch_data_x, encoder_DP, decoder_DP, diffusion_steps, word2idx, 'multinomial', weight)
            betas = calculate_my_betas(beta_ins, standard_betas)
            betas_ins_list.append(beta_ins)
            betas_list.append(betas)

            # record beta
            if e == 0:
                msg = f'{batch_data_x[:10].tolist(), beta_ins[:10].tolist()}'
                record_information(checkpoint_path, msg, 'train_x_beta_ins')

            # diffusion
            if not timestep_respacing:
                timestep_respacing = [diffusion_steps]

            diffusion = SpacedDiffusion(
                use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
                betas=betas,
                rescale_timesteps=rescale_timesteps,
                predict_xstart=predict_xstart,
                learn_sigmas = learn_sigma,
                sigma_small = sigma_small,
                use_kl = use_kl,
                rescale_learned_sigmas=rescale_learned_sigmas
            )

            # calculate lossD
            modelD_DP.train()
            zero_grad(master_params)

            loop = ((batch_data.shape[0]-1) // micro_batch_size_D)+1
            for i in range(loop):
                t, weights = schedule_sampler.sample(batch_hidden[i*micro_batch_size_D:(i+1)*micro_batch_size_D].shape[0], dev())

                compute_losses = functools.partial(
                    diffusion.training_losses,
                    modelD_DP,
                    batch_hidden[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
                    t.to(dev()),
                    i, # offsets
                    batch_data[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
                    batch_mask[i*micro_batch_size_D:(i+1)*micro_batch_size_D].to(dev()),
                )

                losses = compute_losses()

                if isinstance(schedule_sampler, LossAwareSampler):
                    schedule_sampler.update_with_local_losses(
                        t, losses["loss"].detach()
                    )

                loss = losses['loss'] * weights
                loss= loss.mean()

                # output loss calculate
                if not th.isnan(loss):
                    name2val = name2val * name2cnt / (name2cnt + 1) + loss.item() / (name2cnt + 1)
                    name2cnt = name2cnt + 1
                else:
                    write_log(f'[Warning] loss nan')

                loss.backward()
                del losses, loss, t, weights

        # _anneal_lr
        frac_done = (epoch + 1) / epochs
        lr = learning_rate * (1 - frac_done)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # update model D
        optimizer.step()
        # update ema_params
        update_ema(ema_params, master_params, rate=ema_rate)
        write_log(f'[Model D exploration] loss: {name2val:.4f}')

        lossB = 0
        for e in range(exploration):
            write_log(f'[Exploration 2] {e+1} / {exploration}')
            t, weights = schedule_sampler.sample(batch_data.shape[0], dev())
            reward = get_loss_reward(
                betas_list[e], 
                batch_data.to(dev()), 
                batch_mask.to(dev()),
                modelD_DP,
                diffusion_steps,
                t,
                0, # offets
                timestep_respacing, 
                rescale_timesteps,
                predict_xstart,
                learn_sigma,
                use_kl,
                rescale_learned_sigmas,
            )

            meta_reward = (reward - reward_base).to(dev())
            meta_reward = meta_reward[:,0]*weights

            meta_reward = th.where(meta_reward>=0, 1, -1)

            # record meta reward
            if e == 0:
                msg = f'{meta_reward[:10].tolist()}'
                record_information(checkpoint_path, msg, 'meta_rewards')

            meta_rewards = th.zeros(len(batch_data), diffusion_steps)
            meta_rewards[i][t[i]] = meta_reward[i]

            # calculate lossB
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_DP.train()
            decoder_DP.train()

            loss = 0
            loop = ((batch_data.shape[0]-1) // micro_batch_size_B)+1
            for i in range(loop):
                encoder_output, encoder_h, encoder_c = encoder_DP(batch_data_x[i*micro_batch_size_B:(i+1)*micro_batch_size_B].to(dev()))
                decoder_input = th.cat((th.tensor([[word2idx['BOS']]] * len(betas_ins_list[e][i*micro_batch_size_B:(i+1)*micro_batch_size_B])), betas_ins_list[e][i*micro_batch_size_B:(i+1)*micro_batch_size_B]), dim=1)
                decoder_output = decoder_DP(decoder_input.to(dev()), encoder_h, encoder_c)
                decoder_output = th.log(decoder_output)
                for j in range(decoder_output.shape[0]):
                    loss += criterion(decoder_output[j][:-1], betas_ins_list[e][i*micro_batch_size_B:(i+1)*micro_batch_size_B][j].to(dev()))*meta_rewards[i*micro_batch_size_B:(i+1)*micro_batch_size_B][j, :-1].to(dev())
            loss = th.sum(loss)
            lossB += loss

            loss.backward()

        # update model B
        encoder_optimizer.step()
        decoder_optimizer.step()
        write_log(f'[Model B exploration] loss: {lossB / exploration:.4f}')

    update_D_count = (update_D_count + 1) % update_D_loop

    memory_usage = get_gpu_memory()
    write_log(f'[Memory] Memory usage: {memory_usage}')
    write_log(f'[Memory] CPU memory usage: {psutil.virtual_memory()[3] / (1024**3):.6f} G')

    if (epoch+1) % save_interval == 0:
        write_log(f'[Save] Save model D intervally.')
        save_model(encoder, 'encoder', encoder_params, epoch+1)
        save_model(decoder, 'decoder', decoder_params, epoch+1)
        save_model(modelD, 'modelD', master_params, epoch+1)
        save_model(modelD, 'modelD_ema', ema_params, epoch+1)

    elapsed_time = time.time() - start_time
    write_log(f'[Elapsed time] {elapsed_time:.4f} secs.')
    del batch_hidden, batch, batch_data, batch_mask
    gc.collect()


# In[ ]:




