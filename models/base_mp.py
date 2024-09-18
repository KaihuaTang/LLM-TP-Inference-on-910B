
import math
from typing import List, Optional, Tuple, Union
import torch
try:
    import torch_npu
except:
    print('NPU is not detected')
from torch import nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseParallelLayer(torch.nn.Module):
    def __init__(self):
        super(BaseParallelLayer, self).__init__()
        # we need this abstract class to seperate normal layers and parallel layers
        
    def set_dist_info(self, world_size, dist, rank, use_mc2=False, hcomm_info=None):
        self.world_size = world_size
        self.dist = dist
        self.rank = rank
        self.use_mc2 = use_mc2
        self.hcomm_info = hcomm_info
        
    def all_gather(self, x, dim):
        x = x.contiguous()
        # Prepare a list to gather tensors from all processes
        gather_list = [x.clone() for _ in range(self.world_size)]
        self.dist.all_gather(gather_list, x)
        output = torch.cat(gather_list, dim=dim)
        return output
    
    def split(self, x, dim):
        local_size = x.shape[dim] // self.world_size
        output = torch.split(x, local_size, dim=dim)[self.rank]
        return output
    
    def all_reduce(self, x):
        self.dist.all_reduce(x, op=self.dist.ReduceOp.SUM)
        return x
        
    def reduce_scatter(self, x, dim):
        # all reduce
        self.dist.all_reduce(x, op=self.dist.ReduceOp.SUM)
        # split
        local_size = x.shape[dim] // self.world_size
        output = torch.split(x, local_size, dim=dim)[self.rank]
        return output
        
        
    @abstractmethod
    def turn_global_weights_to_local(self):
        # this method has to be implemented
        # so we can get local nn.parameters
        pass
    



class ColumnParallelLinear(BaseParallelLayer):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_input: bool = False,
        gather_output: bool = False,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.gather_input = gather_input
        self.gather_output = gather_output
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.bias = None

    def turn_global_weights_to_local(self):
        local_dim = self.out_features // self.world_size
        local_weight = nn.Parameter(torch.Tensor(local_dim, self.in_features))
        with torch.no_grad():  # Ensure no gradients are tracked during this operation
            local_weight.copy_(torch.split(self.weight, local_dim, dim=0)[self.rank])
        self.weight = local_weight
        
        if self.bias is not None:
            local_bias = nn.Parameter(torch.Tensor(local_dim))
            with torch.no_grad():  
                local_bias.copy_(torch.split(self.bias, local_dim, dim=0)[self.rank])
            self.bias = local_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.gather_input:
            # Prepare a list to gather tensors from all processes            
            input_parallel = self.all_gather(input, dim=-1)
        else:
            input_parallel = input
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = self.all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, gather_input={}, gather_output={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.gather_input, self.gather_output
        )
        
        
        
class RowParallelLinear(BaseParallelLayer):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scatter_input: bool = False,
        scatter_output: bool = False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.scatter_input = scatter_input
        self.scatter_output = scatter_output

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.bias = None


    def turn_global_weights_to_local(self):
        local_dim = self.in_features // self.world_size
        
        if self.use_mc2:
            # mc2 need manually transpose weight
            local_weight = nn.Parameter(torch.Tensor(local_dim, self.out_features))
            with torch.no_grad():  # Ensure no gradients are tracked during this operation
                local_weight.copy_(torch.transpose(torch.split(self.weight, local_dim, dim=1)[self.rank], 0, 1))
            #print(f'in_dim {self.in_features}, out_dim {self.out_features}, local dim {local_dim}, bias {self.bias}')
        else:
            local_weight = nn.Parameter(torch.Tensor(self.out_features, local_dim))
            with torch.no_grad():  # Ensure no gradients are tracked during this operation
                local_weight.copy_(torch.split(self.weight, local_dim, dim=1)[self.rank])
        self.weight = local_weight
        
        if (self.bias is not None):
            local_bias = nn.Parameter(torch.Tensor(self.out_features))
            with torch.no_grad():  
                local_bias.copy_(self.bias / self.world_size)
            self.bias = local_bias
            

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type:ignore
        # Set up backprop all-reduce.
        if self.scatter_input:
            input_parallel = self.split(input, dim=-1)
        else:
            input_parallel = input


        if self.scatter_output:
            # Matrix multiply.
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            # Reduce-scatter across all the partitions.
            output = self.reduce_scatter(output_parallel, dim=-1)
        elif self.use_mc2: 
            output = torch_npu.npu_mm_all_reduce_base(x1=input_parallel,
                                                      x2=self.weight,
                                                      hcom=self.hcomm_info,
                                                      reduce_op="sum",
                                                      bias=self.bias,
                                                      antiquant_scale=None,
                                                      antiquant_offset=None,
                                                      x3=None,
                                                      dequant_scale=None,
                                                      antiquant_group_size=0
                                                      )
        else:
            # Matrix multiply.
            output_parallel = F.linear(input_parallel, self.weight, self.bias)
            # All-reduce across all the partitions.
            output = self.all_reduce(output_parallel)

        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, scatter_input={}, scatter_output={}'.format(
            self.in_features, self.out_features, self.bias is not None, 
            self.scatter_input, self.scatter_output
        )

    
    
    
class QKVColumnParallelLinear(BaseParallelLayer):
    # the linear layer that merges q, k, v linears
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_input: bool = False,
        gather_output: bool = False,
    ) -> None:
        super(QKVColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.gather_input = gather_input
        self.gather_output = gather_output
        
        assert self.out_features % 3 == 0, "The out_dim should be the combination of Q,K,V"
        self.single_outs = self.out_features // 3
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.bias = None

    def turn_global_weights_to_local(self):
        local_dim = self.out_features // self.world_size
        local_weight = nn.Parameter(torch.Tensor(local_dim, self.in_features))
        with torch.no_grad():  # Ensure no gradients are tracked during this operation
            qkv_weights = torch.split(self.weight, self.single_outs, dim=0)
            qkv_local_weights = [torch.split(single_weight, local_dim // 3, dim=0)[self.rank] for single_weight in qkv_weights]
            local_weight.copy_(torch.cat(qkv_local_weights, dim=0))
        self.weight = local_weight
        
        if self.bias is not None:
            local_bias = nn.Parameter(torch.Tensor(local_dim))
            with torch.no_grad():  
                qkv_bias = torch.split(self.bias, self.single_outs, dim=0)
                qkv_local_bias = [torch.split(single_bias, local_dim // 3, dim=0)[self.rank] for single_bias in qkv_bias]
                local_bias.copy_(torch.cat(qkv_local_bias, dim=0))
            self.bias = local_bias

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # type: ignore
        if self.gather_input:
            # Prepare a list to gather tensors from all processes            
            input_parallel = self.all_gather(input, dim=-1)
        else:
            input_parallel = input
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = self.all_gather(output_parallel, dim=-1)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, gather_input={}, gather_output={}'.format(
            self.in_features, self.out_features, self.bias is not None,
            self.gather_input, self.gather_output
        )
