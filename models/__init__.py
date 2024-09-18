# base class
from models.base_mp import BaseParallelLayer

# llama2-7b
from models.llama.llama_mp import LlamaForCausalLM as LlamaForCausalLM_MP
from models.llama.llama_tokenizer import LlamaTokenizer

# qwen
from models.qwen.qwen_mp import QWenLMHeadModel as QwenForCausalLM_MP
