# base class
from models.base_mp import BaseParallelLayer

# llama2-7b
from models.llama.llama_mp import LlamaForCausalLM as LlamaForCausalLM_MP
from models.llama.llama_tokenizer import LlamaTokenizer

# qwen-vl
#from models.qwenvl.qwenvl_mp import QWenLMHeadModel as QwenForCausalLM_MP
#from models.qwenvl.qwenvl_config import QWenConfig
#from models.qwenvl.qwenvl_tokenizer import QWenTokenizer
