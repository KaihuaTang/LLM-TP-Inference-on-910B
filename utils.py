import os
from collections import defaultdict
from typing import List, Dict
import time 
import torch 
import torch.distributed as dist


def get_masks_and_position_ids(seq_len, pad_len):
    # generate mask
    masks = torch.zeros(pad_len, pad_len).float()
    temp_mask = torch.ones(pad_len, pad_len, dtype=torch.bool).tril(diagonal= seq_len - pad_len)
    masks.masked_fill_(temp_mask.logical_not(), float("-inf"))
    # generate position ids
    position_ids = torch.LongTensor([list(range(pad_len))])
    return masks, position_ids
    
    
def encode_pair(get_token, context, continuation):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = get_token(context + continuation)
    context_enc = get_token(context)
    context_enc_len = len(context_enc)
    continuation_enc = whole_enc[context_enc_len:]
    return context_enc, continuation_enc


def get_hcomm_info(world_size, rank, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    print(f"device_{rank} init_process_group success")
    if dist.is_available():
        from torch.distributed.distributed_c10d import _get_default_group
        default_pg = _get_default_group()
    if torch.__version__ > '2.0.1':
        hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    else:
        hcomm_info = default_pg.get_hccl_comm_name(rank)
    print(f"device_{rank} get_hccl_comm_name success")
    return dist, hcomm_info

def load_mp_model_tokenizer(model_type, model_path):
    import models
    if model_type == 'llama2':
        model = models.LlamaForCausalLM_MP.from_pretrained(model_path)
        tokenizer = models.LlamaTokenizer.from_pretrained(model_path)
        return model, tokenizer
    elif model_type == 'qwen':
        from transformers import AutoTokenizer 
        model = models.QwenForCausalLM_MP.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer
    else:
        raise ValueError('No Valid Model')
        
