## 基于910B的huggingface LLM模型的TP部署
由于目前GPU的紧缺，一些国内开发者不得不转向使用910B NPU芯片来训练或部署模型，但由于网上910B的教程比较稀缺，因此本项目主要提供了一套简单的910B多卡推理演示代码，帮助基于NPU的开发者上手。其实只要安装好torch_npu，并将dist的backend调整为hccl，其在模型推理时起来与GPU+torch并无太大区别。

同时你也可以将本项目看作Tensor Parallel部署的极简教程，本项目仅依赖pytorch与transformers就可以部署TP推理。

本项目主要有如下贡献：
- 通过本地精简过的代码，快速理解Tensor Parallel（TP并行）的原理，支持910B与Nvidia GPU。
- 学会使用910B推理开源huggingface的LLM模型(提供LLaMA2与Qwen样例)。
- 学会使用910B进行多卡TP并行推理LLM模型(提供LLaMA2与Qwen样例)。

**如果我的开源项目给您带来了启发，提供一些赞助将对我后续的开源工作有很大的帮助。** 
[支持我的后续开源工作❤️🙏](https://kaihuatang.github.io/donate.html) [(往期支持者)](https://kaihuatang.github.io/supporters.html)

## 安装依赖库
1. 首先需要安装最新的CANN包，类比于CUDA。[(链接)](https://www.hiascend.com/en/software/cann/community)
2. 之后需要安装pytorch与torch_npu，注意torch与torch_npu需要版本一致，请参考如下的链接安装。[(链接)](https://github.com/Ascend/pytorch)
3. 与NPU相关的依赖库只有上面两条，之后正常通过pip安装transformers和其他依赖库即可。

## 下载ckpt
1. 本项目使用的LLaMA2链接为：[https://huggingface.co/meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. 本项目使用的Qwen链接为：[https://huggingface.co/Qwen/Qwen-7B-Chat](https://huggingface.co/Qwen/Qwen-7B-Chat)
   
## 运行脚本
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

## TP并行模型部分代码解释
- TP并行基础知识请参考[(链接)](https://zhuanlan.zhihu.com/p/622212228)
- Transformer block中能进行TP并行的为Attention中的所有线性层和FFN中的所有线性层。其中直接以block输入为输入的线性层需要进行ColumnParallelLinear计算（参考代码models/base_mp.py中的ColumnParallelLinear），作为block输出的线性层需要进行RowParallelLinear计算（参考代码models/base_mp.py中的ColumnParallelLinear）。
- 其中ColumnParallelLinear仅需对每张卡进行权重切分即可自动并行，而RowParallelLinear则需要在并行各卡计算完成后使用all_reduce通信同步特征（已在代码中自动完成）。
- 权重切分的实现在模型加载完权重后，由各个卡对所有需要切分的层执行set_dist_info与turn_global_weights_to_local完成，参考main.py中的实现。
- 一些网络中Attention的QKV三者共用同一个Linear，这需要特殊处理，参考model/base_mp.py中的QKVColumnParallelLinear和model/qwen/qwen_mp.py中的实现。

## 自定义模型适配TP并行
- 首先，需要复制模型文件至本地。参考models下的llama与qwen。
- 其次在模型文件中找到Attention与MLP对应的模块，让其继承models/base_mp.py中的BaseParallelLayer。
- 对于Attention，需要将QKV的Linear替换为QKVColumnParallelLinear或三个ColumnParallelLinear，其中gather_input=False，gather_output=False。而attention的output_project则使用RowParallelLinear，其中scatter_input=False，scatter_output=False。同时需要手动定义turn_global_weights_to_local函数，将模型中所有涉及中间维度的模块参数除以world_size（TP并行的GPU/NPU个数），一般在Attention中仅需修改num_head为local_head=num_head//world_size即可。
- 对于MLP，需要将以input作为输入的模块修改为ColumnParallelLinear，其中gather_input=False，gather_output=False。将输出的模块修改为RowParallelLinear其中scatter_input=False，scatter_output=False。由于大部分模型的MLP代码会自动适应中间维度的变化，因此turn_global_weights_to_local只需pass即可（但不可缺少）。
- 最后在utils.py中的load_mp_model_tokenizer增加自定义模型的载入即可，也可直接在main.py中修改载入和推理代码。

## 引用
如果您发现此项目对您的研究有所帮助，请考虑在您的论文中引用我们的项目。
```
@misc{tang2024tp4910b,
    title = {LLM Tensor Parallel Inference on 910B},
    author = {Tang, Kaihua},
    year = {2024},
    note = {\url{https://github.com/KaihuaTang/LLM-TP-Inference-on-910B}},
}
```
