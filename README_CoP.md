# CoP

The official implementation of our paper "[CoP: Agentic Red-teaming for Large Language Models using Composition of Principles]"


---


## Abstract

Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human red-teamers provide a set of red-teaming principles as instructions to an AI agent to find effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known attack success rate by up to 22 times.




## Quick Start

- **Build environment**
```shell
cd CoP_new
pip install -r requirements.txt
```
### Other Requirements

To run CoP algorithm, you will need the access tokens of [OpenAI_API](https://openai.com/index/openai-api/) and [Grok-2](https://x.ai/api).

As well as the GPU of:
* NVIDIA A800
* VRAM: 80GB

## 🌴 CoP finding the jailbreak prompt
- **Run**\
  *We use Grok-2-1212 (from their official API) as the foundation model for the Red-Teaming Agent. We utilize OpenAI's GPT-4 to act as a Judge LLM. In the following test samples we set the Target LLM as [Llama-2-7B-Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)* 

### Before Training
1. Download the Llama-2-7B-Chat and change the data file path to your local directory path in the following python files
```
main.py
config.py
```
2. Get Access to the GROK-2 API and OpenAI API and put the API key in the following file:
```
language_models_sonnet.py (Line 98 & 114 (for GPT); Line 450 (for Grok))

reward_helper.py (Line 40 & 56)

```
### Finding jailbreak prompt with CoP
To run CoP with Llama-2-7B-Chat model use the following command:

```shell 
python main.py 
```
The results will be saved in the folder:
```
/YOUR_LOCAL_PATH/llama-2_test_score_10_harmbench_saved_all_grok
```

### Evaluating the performance
To evaluate the performance:
1. Download the Harmbench Classifier [HERE](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)
2. Change the path to your local path in
```
evaluate.py
```
3. Run
```shell 
python evaluate.py 
```



