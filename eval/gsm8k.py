import torch
import re
import os
import random
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset, Dataset

from peft import PeftModel, PeftConfig
import argparse
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import json
import time
from dataclasses import asdict, field
from trl_trainer.grpo_trainer import TensorLoRARequest
from safetensors.torch import load_file as safe_load_file

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions() -> Dataset:
    data = load_dataset('openai/gsm8k', 'main', split="test") # type: ignore
    system_prompt = """
        Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: 
        <think>
        ...
        </think>
        <answer>
        ...
        </answer>
    """
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def get_amc_questions() -> Dataset:
    data = load_dataset('math-ai/amc23', split="test") # type: ignore
    system_prompt = """
        Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: 
        <think>
        ...
        </think>
        <answer>
        ...
        </answer>
    """
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x['answer']
    }) # type: ignore
    return data # type: ignore

def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")

    lora_config = os.path.join(model_name_or_path, "adapter_config.json")
    if "checkpoint" in model_name_or_path and os.path.isfile(lora_config):
        with open(lora_config, "r", encoding="utf-8") as f:
            config_peft = json.load(f)
        base_path = config_peft["base_model_name_or_path"]
        base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(base_path)

        model_to_merge = PeftModel.from_pretrained(base_model, model_name_or_path, torch_dtype=torch.float16)
        merged_model = model_to_merge.merge_and_unload()
        merged_model.save_pretrained(os.path.join(model_name_or_path, "merged_model"))
        print(f"save merged model to {os.path.join(model_name_or_path, 'merged_model')}")
        tokenizer.save_pretrained(os.path.join(model_name_or_path, "merged_model"))
        # 清除显存占用的对象
        del base_model
        del tokenizer
        del model_to_merge
        del merged_model

        # 释放显存
        torch.cuda.empty_cache()
        model_name_or_path = os.path.join(model_name_or_path, "merged_model")


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=8, gpu_memory_utilization=0.7)  # 替换成本地路径


    return llm, tokenizer

def load_nvfp4(model_name_or_path):
    """
    Load model with LoRA adapter without reloading base model weights.
    Directly reads adapter weights from checkpoint files.
    """
    print(f"Loading model from {model_name_or_path} ...")

    lora_config_path = os.path.join(model_name_or_path, "adapter_config.json")
    has_lora = "checkpoint" in model_name_or_path and os.path.isfile(lora_config_path)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if has_lora:
        # Load PEFT configuration
        peft_config = PeftConfig.from_pretrained(model_name_or_path)
        base_path = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        
        # Directly load adapter weights from file without loading base model
        # Try safetensors first, fall back to pytorch
        adapter_path = os.path.join(model_name_or_path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            print(f"Loading adapter weights from {adapter_path}")
            peft_params = safe_load_file(adapter_path)
        else:
            adapter_path = os.path.join(model_name_or_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                print(f"Loading adapter weights from {adapter_path}")
                peft_params = torch.load(adapter_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No adapter weights found in {model_name_or_path}")
        
        # Separate layernorm params from LoRA params
        layernorm_params = {}
        lora_params = {}
        
        for name, param in peft_params.items():
            if 'layernorm' in name or 'modules_to_save' in name:
                # Remove 'base_model.model.' prefix for vLLM
                llm_model_name = name.replace('base_model.model.', '')
                # Also handle modules_to_save prefix
                llm_model_name = llm_model_name.replace('modules_to_save.default.', '')
                layernorm_params[llm_model_name] = param
            else:
                lora_params[name] = param
        
        print(f"Separated {len(layernorm_params)} layernorm params and {len(lora_params)} LoRA params")
        
        # Update peft_config to exclude modules_to_save
        peft_config.modules_to_save = None
        
        # Initialize vLLM with LoRA support (base model is loaded here)
        max_lora_rank = 32
        llm = LLM(
            model=base_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.7,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
        )
        
        # Load layernorm weights into vLLM
        if layernorm_params:
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            weights_to_load = [(name, param.to(torch.float16)) for name, param in layernorm_params.items()]
            llm_model.load_weights(weights_to_load)
            print(f"Loaded {len(layernorm_params)} layernorm parameters to vLLM")
        
        # Create and add LoRA adapter to vLLM
        if lora_params:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_request = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="lora_adapter_path",
                peft_config=asdict(peft_config),
                lora_tensors=lora_params,
            )
            llm.llm_engine.add_lora(lora_request)
            print(f"Added LoRA adapter to vLLM with {len(lora_params)} parameters")
        
    else:
        # No LoRA adapter, load model directly
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        llm = LLM(model=model_name_or_path, tensor_parallel_size=1, gpu_memory_utilization=0.7)
    
    return llm, tokenizer

def generate(llm, tokenizer, input_text, generate_kwargs, args):
    text = tokenizer.apply_chat_template(
        input_text,
        tokenize=False,
        add_generation_prompt=True
    )

    # max_tokens is for the maximum length for generation.
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=1024)
    
    # Check if there are LoRA adapters loaded in vLLM
    if args.load == "nvfp4":
        lora_requests = None
        try:
            # Check if LoRA is enabled by trying to access lora_manager
            model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
            if hasattr(model_runner, 'lora_manager') and model_runner.lora_manager is not None:
                lora_int_ids = list(llm.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    # Use the first LoRA adapter
                    lora_int_id = lora_int_ids[0]
                    lora_requests = LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="lora_stub_path"
                    )
        except (AttributeError, Exception):
            # LoRA is not enabled, continue without LoRA
            pass
        
        # Generate with or without LoRA
        outputs = llm.generate(text, sampling_params, lora_request=lora_requests, use_tqdm=False)
    else:
        outputs = llm.generate(text, sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

def generate_batch(llm, tokenizer, input_texts, generate_kwargs, args):
    """Batch generate for multiple prompts"""
    texts = []
    for input_text in input_texts:
        text = tokenizer.apply_chat_template(
            input_text,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
    
    # max_tokens is for the maximum length for generation.
    if "checkpoint" in args.model or "ckpt" in args.model:
        max_tokens = generate_kwargs.get("max_new_tokens", 4096)
    else:
        max_tokens = generate_kwargs.get("max_new_tokens", 2048)
    
    sampling_params = SamplingParams(
        temperature=generate_kwargs.get("temperature", 0.8),
        top_p=generate_kwargs.get("top_p", 0.95),
        max_tokens=max_tokens
    )
    
    # Check if there are LoRA adapters loaded in vLLM
    if args.load == "nvfp4":
        lora_requests = None
        try:
            # Check if LoRA is enabled by trying to access lora_manager
            model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
            if hasattr(model_runner, 'lora_manager') and model_runner.lora_manager is not None:
                lora_int_ids = list(llm.llm_engine.list_loras())
                if len(lora_int_ids) > 0:
                    # Use the first LoRA adapter
                    lora_int_id = lora_int_ids[0]
                    lora_requests = LoRARequest(
                        lora_name=f"{lora_int_id}",
                        lora_int_id=lora_int_id,
                        lora_path="lora_stub_path"
                    )
        except (AttributeError, Exception):
            # LoRA is not enabled, continue without LoRA
            pass
        
        # Generate with or without LoRA
        outputs = llm.generate(texts, sampling_params, lora_request=lora_requests, use_tqdm=True)
    else:
        outputs = llm.generate(texts, sampling_params, use_tqdm=True)
    
    # Extract texts from outputs
    return [output.outputs[0].text for output in outputs]

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward_func(prompts, response, answer, **kwargs) -> list[float]:
    q = prompts
    extracted_responses = extract_xml_answer(response)
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer}", f"\nResponse:\n{response}", f"\nExtracted:\n{extracted_responses}")
    return [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)] or [0.0]

def accuracy_reward(prompts, completions, answer):
    """Reward function that checks if the completion is the same as the ground truth."""
    rewards = []
    extracted_responses = extract_xml_answer(completions)
    if extracted_responses == answer:
        return [1]

    for content, sol in zip([completions], [answer]):
        print(content, sol)
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = 0
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)
    print(rewards)
    print('-'*20, f"Question:\n{prompts}", f"\nResponse:\n{completions}", f"\nAnswer:\n{gold_parsed}", f"\Extract:\n{answer_parsed}",type(answer_parsed))

    return rewards

def math_verified(prompts, response, answer, **kwargs) -> list[float]:
    q = prompts
    # extracted_responses = extract_xml_answer(response)
    return accuracy_reward(prompts,response, answer)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="/lustre/fsw/portfolios/nvr/users/weihua/model/qwen2.5-32B-Instruct-bigmath_awq-nvfp4-fake-ckpt_290",
        help="The model checkpoint for weights initialization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="amc",
    )

    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.data == "gsm8k":
        data = get_gsm8k_questions()
    else:
        data = get_amc_questions()

    if args.load == "nvfp4":
        llm, tokenizer = load_nvfp4(args.model)
    else:
        llm, tokenizer = load(args.model)

    # Prepare all prompts and answers for batch processing
    input_texts = []
    prompts = []
    gold_answers = []
    
    for sample in data:
        input_texts.append(sample["prompt"])
        if len(sample["prompt"]) > 1:
            prompts.append(sample["prompt"][1]['content'])
        else:
            prompts.append(sample["prompt"][0]['content'])
        gold_answers.append(sample["answer"])
    
    # Prepare generate_kwargs
    if "checkpoint" in args.model or "ckpt" in args.model:
        generate_kwargs = dict(max_new_tokens=4096, top_p=0.95, temperature=0.8)
    else:
        generate_kwargs = dict(max_new_tokens=2048, top_p=0.95, temperature=0.8)
    
    # Batch generate all responses
    print(f"Generating responses for {len(input_texts)} samples in batch...")
    responses = generate_batch(llm, tokenizer, input_texts, generate_kwargs, args)
    
    # Process all results
    answers = []
    for prompt, response, gold_answer in tqdm(zip(prompts, responses, gold_answers), total=len(prompts), desc="Evaluating"):
        if args.data == "gsm8k":
            answers.append(correctness_reward_func(prompt, response, gold_answer)[0])
        else:
            answers.append(math_verified(prompt, response, gold_answer)[0])
    
    print(
        f"Num of total question: {len(answers)}, "
        f"Correct num: {sum(answers)}, "
        f"Accuracy: {float(sum(answers))/len(answers)}."
    )

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, f"{args.model}_amc_results.txt"), "w") as f:
        # Write each answer on a new line
        # Write the summary statistics
        f.write(
            f"Num of total questions: {len(answers)}\n"
            f"Correct num: {sum(answers)}\n"
            f"Accuracy: {float(sum(answers)) / len(answers):.2f}\n"
        )


if __name__ == "__main__":
    main()