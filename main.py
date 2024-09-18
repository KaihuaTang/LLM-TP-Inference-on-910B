import torch
# check whether NPU (910B) is available
try:
    import torch_npu
    USE_NPU = True
except:
    USE_NPU = False
import argparse
import os
import yaml
import math
import random
from datetime import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Process
import torch.distributed as dist

import models
from utils import load_mp_model_tokenizer, get_hcomm_info

def run_multi_npu(args, rank, port, user_input):
    # init dist
    torch_npu.npu.set_device(rank)
    print("current device:", torch_npu.npu.current_device())
    print('[INFO] device_{} 创建HCCL通信链路 '.format(rank))
    dist, hcomm_info = get_hcomm_info(int(args.world_size), rank, port)

    # load model
    if rank == 0:
        print(f"Load {args.model_type} from {args.model_path}")
    model, tokenizer = load_mp_model_tokenizer(args.model_type, args.model_path)
    
    # initialize tensor parallel
    if rank == 0:
        print(f"Apply Tensor Parallel: start spliting weights to each device")
    for name, m in model.named_modules():
        if isinstance(m, models.BaseParallelLayer):
            m.set_dist_info(args.world_size, dist, rank, args.use_mc2, hcomm_info)
            m.turn_global_weights_to_local()
            if rank == 0:
                print(f"==> Split weights of layer {name} to each device (model parallel). [MC2 Switch: {args.use_mc2}]")
            
    model = model.bfloat16().npu(rank)
    model = model.eval()
    
    if args.model_type == 'llama2':
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        input_ids = input_ids.npu()
        output = model.generate(input_ids, max_length=256)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    elif args.model_type == 'qwen':
        response, _ = model.chat(tokenizer, user_input, history=None)
        
    if rank == 0:
        print("=====================================================================")
        print(f"Response: {response}")
        return response
    else:
        return None


def run_multi_gpu(args, rank, port, user_input):
    # init dist
    print("current device:", torch.cuda.current_device())
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=int(args.world_size))
    print(f"device_{rank} init_process_group success")

    # load model
    if rank == 0:
        print(f"Load {args.model_type} from {args.model_path}")
    model, tokenizer = load_mp_model_tokenizer(args.model_type, args.model_path)
    
    # initialize tensor parallel
    if rank == 0:
        print(f"Apply Tensor Parallel: start spliting weights to each device")
    for name, m in model.named_modules():
        if isinstance(m, models.BaseParallelLayer):
            m.set_dist_info(args.world_size, dist, rank, False)
            m.turn_global_weights_to_local()
            if rank == 0:
                print(f"==> Split weights of layer {name} to each device (model parallel). [MC2 Switch: {args.use_mc2}]")
            
    model = model.bfloat16().cuda(rank)
    model = model.eval()
    
    if args.model_type == 'llama2':
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        input_ids = input_ids.cuda(rank)
        output = model.generate(input_ids, max_length=256)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
    elif args.model_type == 'qwen':
        response, _ = model.chat(tokenizer, user_input, history=None)
        
    if rank == 0:
        print("=====================================================================")
        print(f"Response: {response}")
        return response
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--use_mc2', default=False, action='store_true')
    args = parser.parse_args()

    if USE_NPU:
        print("Detect NPU(910B) environment.")
        target_func = run_multi_npu
    elif torch.cuda.is_available():
        print("Detect GPU environment")
        target_func = run_multi_gpu
    else:
        raise ValueError("Neither NPU nor GPU environment is given!")
    
    user_input = input("Please ask something: ")

    p_list = []
    port = 30000 + random.randint(0, 10000)
    process_num = int(args.world_size)
    mp.set_start_method("forkserver", force=True)
    for rank in range(process_num):
        p = Process(target=target_func, args=(args, rank, port, user_input))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    
if __name__=='__main__':
    main()
