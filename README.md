# Understanding-Tensor-Parallel-Using-910B
由于目前GPU的紧缺，一些国内开发者不得不转向使用910B NPU芯片来训练或部署模型，但由于网上910B的教程比较稀缺，因此本项目主要提供了一套简单的910B多卡推理演示代码，帮助基于NPU的开发者上手。其实只要安装好torch_npu，并将dist的backend调整为hccl，其在模型推理时起来与GPU+torch并无太大区别。本项目主要有如下贡献：
- 通过本地精简过的代码，快速理解Tensor Parallel（TP并行）的原理。
- 学会使用910B推理开源huggingface的LLM模型(提供LLaMA2与Qwen样例)。
- 学会使用910B进行多卡TP并行推理LLM模型(提供LLaMA2与Qwen样例)。

# 安装依赖库
更新中

# 运行脚本
python main.py --model_type llama2 --model_path /home/ma-user/work/t00831955/checkpoints/llama-2-7b-hf --world_size 8 --use_mc2

# TP并行原理
