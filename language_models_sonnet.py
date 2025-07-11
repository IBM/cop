import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import vertexai
import random
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
# from vllm import LLM
# from vllm.sampling_params import SamplingParams
from config import MINISTRAL_8B_PATH
import requests

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
    
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    api_key = "YOUR-API-KEY"

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        client = openai.OpenAI(api_key="YOUR-API-KEY")
        for _ in range(self.API_MAX_RETRY):
            try:
                response = client.chat.completions.create(
                            model = self.model_name,
                            messages = conv,
                            max_tokens = max_n_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            #request_timeout = self.API_TIMEOUT,
                            )
                output = response.choices[0].message.content
                break
            except openai.OpenAIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = "YOUR-API-KEY"
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.messages.create(
                    model="claude-2.1",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": conv}
                    ]
                )
                output = completion.content[0].text
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Claude3():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = "YOUR-API-KEY"
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                print(conv)
                completion = self.model.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": conv}
                    ]
                )
                output = completion.content[0].text
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Gemini():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:
        PROJECT_ID = "directed-galaxy-440602-s3"  # @param {type: "string", placeholder: "[your-project-id]" isTemplate: true}
        if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
            PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        
        LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/root/.config/gcloud/application_default_credentials.json"
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        MODEL_ID = "gemini-1.5-pro-002"  # @param {type:"string"}

        self.model = GenerativeModel(MODEL_ID)

        self.generation_config = GenerationConfig(
            temperature=0.9,
            top_p=1.0,
            top_k=32,
            candidate_count=1,
            max_output_tokens=8192,
        )
        
        # Set safety settings
    
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.OFF,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.OFF,
        }
    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                completion = self.model.generate_content(
                    conv,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                )
                output = completion.text
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Ministral():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        model_name = MINISTRAL_8B_PATH

        self.sampling_params = SamplingParams(max_tokens=8192)

        self.model = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": conv
                    },
                ]
                completion = self.model.chat(messages, sampling_params=self.sampling_params, tensor_parallel_size=2)
                output = completion[0].outputs[0].text
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]



class YI():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.model_name = "yi-lightning"

        self.API_BASE = "https://api.lingyiwanwu.com/v1"
        self.API_KEY = "YOUR-API-KEY"

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                template = [{"role": "user", "content": conv}]
                client = openai.OpenAI(
                    api_key=self.API_KEY,
                    base_url=self.API_BASE
                )
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=template
                )
                output = completion.choices[0].message.content
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class GROK():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.model_name = "grok-2-1212"
        self.url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "YOUR-API-KEY"
        }

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": conv
                        }
                    ],
                    "model": self.model_name,
                    "stream": False,
                    "temperature": 0
                }
                response = requests.post(self.url, headers=self.headers, json=data)
                output = response.json()['choices'][0]['message']['content']
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class DEEPSEEK_CHAT():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    def __init__(self, model_name) -> None:

        self.client = openai.OpenAI(api_key="YOUR-API-KEY", base_url="https://api.deepseek.com")

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        backoff_factor = 2
        for attempt in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": conv},
                    ],
                    stream=False
                )
                output = response.choices[0].message.content
                break
            except Exception as e:
                
                print(f"Request failed: {e}")
                wait_time = backoff_factor * (2 ** attempt) + random.random()
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        palm.configure(api_key=self.API_KEY)

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = palm.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last
                
                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)]
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(1)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
