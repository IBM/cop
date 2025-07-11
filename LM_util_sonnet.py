import random
from fastchat.model import get_conversation_template
from language_models_sonnet import GPT, Gemini, Claude, Claude3, PaLM, HuggingFace, Ministral, GROK, YI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, VICUNA_13B_PATH, LLAMA_PATH, LLAMA_13B_PATH, LLAMA_3_PATH, GEMMA_PATH, BAICHUAN_PATH, BAICHUAN_13B_PATH, QWEN_7B_PATH, QWEN_14B_PATH, MINISTRAL_8B_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P   

import ast
import logging
import regex as re
import json

# def extract_json(text):
#     try:
#         # Use regex to find the JSON block
#         match = re.search(r"```json\n(.*?)```", text, re.DOTALL)
#         if match:
#             json_string = match.group(1)
#             # Parse the JSON string
#             data = json.loads(json_string)
#             keys = list(data.keys())
#             if not all(x in data for x in keys):
#                 logging.error("Error in extracted structure. Missing keys.")
#                 logging.error(f"Extracted:\n {data}")
#                 return None, None, None, None, None
#             new_jb_prompt = data[keys[0]]
#             ops = data[keys[1]]
#             policy = data[keys[2]]
#             return data, text, new_jb_prompt, ops, policy
#         else:
#             return None, None, None, None, None
#     except json.JSONDecodeError:
#         return None, None, None, None, None

def extract_json(s):
    try:
        parsed = re.sub(r"^```\w*\n|\n```$", "", s)
        try:
            parsed = eval(parsed)
        except:
            return None, None, None, None, None
        keys = list(parsed.keys())
        if not all(x in parsed for x in keys):
            logging.error("Error in extracted structure. Missing keys.")
            logging.error(f"Extracted:\n {parsed}")
            return None, None, None, None, None
        new_jb_prompt = parsed[keys[0]]
        ops = parsed[keys[1]]
        policy = parsed[keys[2]]
        return parsed, s, new_jb_prompt, ops, policy
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {s}")
        return None, None, None, None, None
        
def extract_json_backup(s):
    try:
        json_match = re.search(r'{.*}', s, re.DOTALL)
        if json_match:
            json_like_content = json_match.group(0)
            clean_content = json_like_content.replace("```python", "").replace("```", "").replace("\n", " ").strip()
            parsed = json.loads(clean_content)
            keys = list(parsed.keys())
            if not all(x in parsed for x in keys):
                logging.error("Error in extracted structure. Missing keys.")
                logging.error(f"Extracted:\n {parsed}")
                return None, None, None, None, None
            new_jb_prompt = parsed[keys[0]]
            ops = parsed[keys[1]]
            policy = parsed[keys[2]]
            return parsed, s, new_jb_prompt, ops, policy
        else:
            print("No JSON-like content found.")
            return None, None, None, None, None
        
    except (SyntaxError, ValueError):
        logging.error("Error parsing extracted structure")
        logging.error(f"Extracted:\n {s}")
        return None, None, None, None, None

def conv_template(template_name):
    template = get_conversation_template(template_name)
    if template.name == 'llama-2':
        template.sep2 = template.sep2.strip()
    return template

def load_target_model(args):
    preloaded_model = None
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return targetLM

def load_policy_model(args):
    preloaded_model = None
    policyLM = PolicyLM(model_name = args.helper_model, 
                        max_n_tokens = args.helper_max_n_tokens,
                        max_n_attack_attempts = args.max_n_attack_attempts,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return policyLM


class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.system_message=""
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            elif "claude-2" in self.model_name:
                full_prompts.append(prompt)
            elif "claude-3" in self.model_name:
                full_prompts.append(prompt)
            elif "gemini" in self.model_name:
                full_prompts.append(prompt)
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        outputs_list = self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
        return outputs_list




class PolicyLM():
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            max_n_attack_attempts: int,
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_prompt(self, attack_prompt, action_type):
        return f"{attack_prompt}"
        # if action_type =="restart":
        #     return f""
        # else:
        #     return f"{attack_prompt}"

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        indices_to_regenerate = list(range(batchsize))
        convs_list = [conv_template(self.template) for _ in range(batchsize)]
        valid_options = [None] * batchsize
        valid_policy = [None] * batchsize
        full_prompts = []
        for attempt in range(self.max_n_attack_attempts):
            for conv, prompt in zip(convs_list, prompts_list):
                conv.system_message=""
                conv.append_message(conv.roles[0], prompt)
                if "gpt" in self.model_name:
                    # Openai does not have separators
                    full_prompts.append(conv.to_openai_api_messages())
                elif "palm" in self.model_name:
                    full_prompts.append(conv.messages[-1][1])
                elif "ministral" in self.model_name:
                    full_prompts.append(conv.to_openai_api_messages())
                elif "grok" in self.model_name:
                    full_prompts.append(prompt)
                elif "yi" in self.model_name:
                    full_prompts.append(prompt)
                else:
                    conv.append_message(conv.roles[1], None) 
                    full_prompts.append(conv.get_prompt())
            outputs_list = self.model.batched_generate(full_prompts, 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = 1,
                                                            top_p = 0.9
                                                        )
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                try:
                    orig_index = indices_to_regenerate[i]
                except:
                    print("##############ERROR###########")
                    print(indices_to_regenerate)
                    print(i)
                    print(full_output)
                attack_dict, json_str, jb_goal, jb_options, jb_policy = extract_json_backup(full_output)
                if any(var is None for var in [attack_dict, json_str, jb_goal, jb_options, jb_policy]):
                    attack_dict, json_str, jb_goal, jb_options, jb_policy = extract_json(full_output)
                
                if attack_dict is not None:
                    valid_options[orig_index] = jb_options
                    valid_policy[orig_index] = jb_policy
                    #convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        return valid_options, valid_policy


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name in ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"]:
        lm = GPT(model_name)
    elif model_name in ["gemini"]:
        lm = Gemini(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    elif model_name in ["claude-3"]:
        lm = Claude3(model_name)
    elif model_name in ["palm-2"]:
        lm = PaLM(model_name)
    elif model_name in ["ministral"]:
        model_name = MINISTRAL_8B_PATH
        lm = Ministral(model_name)
    elif model_name in ["grok"]:
        lm = GROK(model_name)
    elif model_name in ["yi"]:
        lm = YI(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=True
        ) 

        if 'llama-2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4-1106-preview":{
            "path":"gpt-4-1106-preview",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "vicuna-13b":{
            "path":VICUNA_13B_PATH,
            "template":"vicuna_v1.1"
        },
        "llama-2":{
            "path":LLAMA_PATH,
            "template":"llama-2"
        },
        "llama-2-13b":{
            "path":LLAMA_13B_PATH,
            "template":"llama-2"
        },
        "llama-3":{
            "path":LLAMA_3_PATH,
            "template":"llama-3"
        },
        "gemma":{
            "path":GEMMA_PATH,
            "template":"gemma"
        },
        "baichuan-7b":{
            "path":BAICHUAN_PATH,
            "template":"baichuan2-chat"
        },
        "baichuan-13b":{
            "path":BAICHUAN_13B_PATH,
            "template":"baichuan2-chat"
        },
        "qwen-7b":{
            "path":QWEN_7B_PATH,
            "template":"qwen-7b-chat"
        },
        "qwen-14b":{
            "path":QWEN_14B_PATH,
            "template":"qwen-14b-chat"
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "claude-3":{
            "path":"claude-3",
            "template":"claude-3"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        },
        "gemini":{
            "path":"gemini-pro",
            "template":"gemini-1.5-pro"
        },
        "ministral":{
            "path":MINISTRAL_8B_PATH,
            "template":"mistral"
        },
        "grok":{
            "path": "grok-2-1212",
            "template": "gpt-4"
        },
        "yi":{
            "path": "yi-lightning",
            "template": "gpt-4"
        }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template



    