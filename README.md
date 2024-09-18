# 基于910B的huggingface LLM模型的TP部署
由于目前GPU的紧缺，一些国内开发者不得不转向使用910B NPU芯片来训练或部署模型，但由于网上910B的教程比较稀缺，因此本项目主要提供了一套简单的910B多卡推理演示代码，帮助基于NPU的开发者上手。其实只要安装好torch_npu，并将dist的backend调整为hccl，其在模型推理时起来与GPU+torch并无太大区别。

同时你也可以将本项目看作Tensor Parallel部署的极简教程，本项目仅依赖pytorch与transformers就可以部署TP推理。

本项目主要有如下贡献：
- 通过本地精简过的代码，快速理解Tensor Parallel（TP并行）的原理，支持910B与Nvidia GPU。
- 学会使用910B推理开源huggingface的LLM模型(提供LLaMA2与Qwen样例)。
- 学会使用910B进行多卡TP并行推理LLM模型(提供LLaMA2与Qwen样例)。

# 安装依赖库
1. 首先需要安装最新的CANN包，类比于CUDA。[(链接)](https://www.hiascend.com/en/software/cann/community)
2. 之后需要安装pytorch与torch_npu，注意torch与torch_npu需要版本一致，请参考如下的链接安装。[(链接)](https://github.com/Ascend/pytorch)
3. 与NPU相关的依赖库只有上面两条，之后正常通过pip安装transformers和其他依赖库即可。

# 下载ckpt
1. 本项目使用的LLaMA2链接为：[https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. 本项目使用的Qwen链接为：[https://huggingface.co/Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
   
# 运行脚本
```
# RUN llama-2-7b-hf
python main.py --model_type llama2 --model_path /home/ma-user/work/t00831955/checkpoints/llama-2-7b-hf --world_size 8 --use_mc2

# RUN Qwen-7B-Chat
python main.py --model_type qwen --model_path /home/ma-user/work/t00831955/checkpoints/Qwen-7B-Chat --world_size 8 --use_mc2
```
参数解释：
- model_type输入llama2 / qwen。注意，之所以需要指定模型名，是因为本项目的手搓TP并行需要修改模型代码，因此不同模型需要加载不同代码。但修改并不复杂，后续会解释。
- model_path为通过huggingface下载的模型ckpt路径。
- world_size为开启TP多卡并行后，本地所使用的卡数，卡数越多单卡所需的计算越小。
- use_mc2为910B最新版本的加速功能，如果你的版本不支持关掉即可。其原理是在TP时，多卡的Linear计算和all_reduce通信实现通信并行。详细来说就是把Linear的计算切成多个小块，每一块计算时上一块的计算结果同时进行all_reduce通信，通过计算和通信流水降低整体耗时。

# TP并行原理

TODO
- 添加Qwen支持
- 完善README
