import openai
import tiktoken

import asyncio

import anthropic # type: ignore
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient

from together import AsyncTogether, Together


import time

from src.LLMs.settings.config import get_settings
from groq import Groq, AsyncGroq
from ipex_llm.optimize import low_memory_init, load_low_bit
from transformers import LlamaTokenizer
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import platform
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
#settings = get_settings()
#rate_limit = settings.rate_limit 

# API keys
#openai.api_key = settings.openai_api_key
#any_api_key = settings.any_api_key
#mistral_api = settings.mistral_api
#anthropic_api = settings.anthropic_api
#cohere_api = settings.cohere_api
#pplx_api = settings.pplx_api
#together_api = settings.together_api

groq_api = ""

# Models config
models = [
    "gpt-3.5",
    "gpt-4",
    'mixtral',
    'mistral-small',
    'mistral-medium',
    "code-llama",
    "command-R+",
    'pplx',
    'mixtral-8x22',
    "mixtral-groq",
    'llama-3',
    'llama-3-any',
    'llama-3-8B',
    'wizard'
]

models_fullnames = {
    "gpt-3.5": "gpt-3.5-turbo-0125",
    "gpt-4": "gpt-4-turbo-2024-04-09",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-small-old": 'open-mixtral-8x7b',
    "mistral-small": 'mistral-small-latest',
    "mistral-medium": 'mistral-medium-latest',
    "mistral-large": 'mistral-large-latest',
    "code-llama": "codellama/CodeLlama-70b-Instruct-hf",
    "claude-small": "claude-3-haiku-20240307",
    "claude-medium": "claude-3-sonnet-20240229",
    "claude-large": "claude-3-opus-20240229",
    "command-R+": "command-r-plus",
    "pplx": "llama-3-sonar-large-32k-online",
    'mixtral-8x22': "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mixtral-groq": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'llama-3': 'meta-llama/Llama-3-70b-chat-hf',
    'llama-3-any': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-3-8B' : 'meta-llama/Llama-3-8b-chat-hf',
    'wizard': "microsoft/WizardLM-2-8x22B"
}

models_endpoints = {
    "gpt-3.5": "openai",
    "gpt-4": "openai",
    "mixtral": "anyscale",
    "mistral-small-old": 'mistral',
    "mistral-small": 'mistral',
    "mistral-medium": 'mistral',
    "mistral-large": 'mistral',
    "code-llama": "anyscale",
    "claude-small": "anthropic",
    "claude-medium": "anthropic",
    "claude-large": "anthropic",
    "command-R+": "cohere",
    "pplx": "perplexity",
    'mixtral-8x22': 'anyscale',
    "mixtral-groq": 'groq',
    'llama-3': 'together',
    'llama-3-any': 'anyscale',
    'llama-3-8B': 'anyscale',
    'wizard': 'together'
}

token_prices = {
    "gpt-3.5": 0.0015 / 1000,
    "gpt-4": 0.06 / 1000,
    "mixtral": 0.5 / 1000000,
    "mixtral-groq": 0.5 / 1000000,
    "mistral-medium": 5 / 1000000,
}


#class RateLimiter:
#    def __init__(self, calls_per_second=0.25):
#        self.calls_per_second = rate_limit
#        self.semaphore = asyncio.Semaphore(calls_per_second)
#        self.next_call_time = time.time()

#    async def wait_for_rate_limit(self):
#        async with self.semaphore:
#            now = time.time()
#            sleep_time = self.next_call_time - now
#            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)
#            if sleep_time > 0:
#                await asyncio.sleep(sleep_time)
#            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)

def submit_prompt_flex(prompt, model="gpt-3.5", output_json=False):
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    saved_dir = '/home/intel/plischwe/llama-2-ipex-llm-4-bit'
    with low_memory_init(): # Fast and low cost by loading model on meta device
        model = AutoModel.from_pretrained(saved_dir,
                                          torch_dtype="auto",
                                          trust_remote_code=True)
    model = load_low_bit(model, saved_dir) # Load the optimized model

    LLAMA2_PROMPT_FORMAT = """
    [INST] <<SYS>>
    You are a helpful assistant.
    <</SYS>>
    {prompt}[/INST]
    """
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=150)
        #output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

async def a_submit_prompt_flex(prompt, model="gpt-3.5", output_json=False):
    # Load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)

    saved_dir = '/home/intel/plischwe/llama-2-ipex-llm-4-bit'
    with low_memory_init(): # Fast and low cost by loading model on meta device
        model = AutoModel.from_pretrained(saved_dir,
                                          torch_dtype="auto",
                                          trust_remote_code=True)
    model = load_low_bit(model, saved_dir) # Load the optimized model

    LLAMA2_PROMPT_FORMAT = """
    [INST] <<SYS>>
    You are a helpful assistant.
    <</SYS>>
    {prompt}[/INST]
    """
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=150)
        #output_str = tokenizer.decode(output[0], skip_special_tokens=True)    
    return output

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embedding(input, dimension=1024):
    #print(f"Input to the embedding function: {input}")
    tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
    model = AutoModel.from_pretrained("thenlper/gte-large")
    tokens = tokenizer(input, max_length=512, padding=True, truncation=True, return_tensors='pt')
    response = model(**tokens)
    embeddings = average_pool(response.last_hidden_state, tokens['attention_mask'])
    embeddings_list = embeddings.tolist()
    #print(f"Response from custom embedding model (should be list of vectors): {embeddings_list}")

    return embeddings_list
