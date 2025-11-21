import logging
from datasets import load_dataset, Dataset


logger = logging.getLogger(__name__)



def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()



# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train[:200]") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    system_prompt = """
        Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: 
        <think>
        ...
        </think>
        <answer>
        ...
        </answer>
    """
    # system_prompt = """
    #     Solve the following math problem step by step. The reasoning process is enclosed within <think> </think> tags, and the final answer is provided after "####", i.e.,: 
    #     <think>
    #     ...
    #     </think>
    #     #### answer
    # """
    # instruction = """
    #     Let\'s think step by step first within <think> </think> tags, and output the final answer after "####" tag, i.e.,: 
    #     #     <think>
    #     #     ...
    #     #     </think>
    #     #     #### number
    # """
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def get_bigmath_questions(data, split = "train") -> Dataset:
    if "hard" in data:
        data = load_dataset('open-r1/Big-Math-RL-Verified-Processed', "level_4_5", split="train")
    else:
        data = load_dataset('open-r1/Big-Math-RL-Verified-Processed', "level_3_4_5", split="train") # type: ignore
    system_prompt = """
        Solve the following math problem step by step. The reasoning process and direct answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>: 
        <think>
        ...
        </think>
        <answer>
        ...
        </answer>
    """
    data = data.filter(lambda x: len(x['prompt']) <= 1024)
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': x['prompt']}
        ],
        'answer': x['solution']
    }) # type: ignore
    return data # type: ignore

    
#        please think with over 512 tokens.

def get_dapo_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the DAPO dataset with optional one-shot prompting."""
    # try:
    #     data = load_dataset("Perflow-Shuai/DAPO-Math-17k",split="train")
    # except Exception as e:
    #     logger.error(f"Failed to load dataset: {e}")
    #     raise
    # # if ultilize the prompt length filter
    # data = data.filter(lambda x: len(x['problem']) <= 1024)
    # def format_example(x):
    #     prompt = [{'role': 'user', 'content': x['problem']}]
    #     return {'prompt': prompt, 'answer': x['answer']}
    try:
        data = load_dataset("open-r1/DAPO-Math-17k-Processed","en", split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    # if ultilize the prompt length filter
    data = data.filter(lambda x: len(x['source_prompt'][0]["content"]) <= 1024)
    def format_example(x):
        prompt = x['source_prompt']
        return {'prompt': prompt, 'answer': x['solution']}

    return data.map(format_example)

def get_mm_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the DAPO dataset with optional one-shot prompting."""
    try:
        data = load_dataset("miromind-ai/MiroMind-M1-RL-62K",split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    # if ultilize the prompt length filter
    data = data.filter(lambda x: len(x['problem']) <= 1024)

    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['clean_answer']
    }) # type: ignore

    return data

def get_code_questions(split="train", use_one_shot=False) -> Dataset:
    """Loads and prepares the DAPO dataset with optional one-shot prompting."""

    # data = load_dataset("open-r1/codeforces", "verifiable-prompts", split="train")
    data = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled", split="train")
    system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer."

    
    def make_conversation(example):
        prompt = []
        prompt.append({"role": "system", "content": system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}
    return data.map(make_conversation)
