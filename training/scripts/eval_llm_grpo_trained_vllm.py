import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import get_peft_config, GRPOConfig, GRPOTrainer, ModelConfig

# Load dataset from Hugging Face Hub
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")

# Dataset({
#     features: ['target', 'nums'],
#     num_rows: 490364
# })
# {'target': 98, 'nums': [44, 19, 35]}

# select a random subset of 50k samples
dataset = dataset.shuffle(seed=42).select(range(50000))

# Load tokenizer from Hugging Face Hub to format the dataset to our "r1" prompt
# model = "meta-llama/Llama-3.2-3B-Instruct"
# model = "Qwen/Qwen2.5-3B-Instruct"
model = "deep-learning-pytorch-huggingface/training/runs/qwen-2.5-3b-r1-countdown/checkpoint-1000"
model = "deep-learning-pytorch-huggingface/training/runs/qwen-2.5-3b-r1-countdown/checkpoint-1700"

model = "meta-llama/Llama-3.2-3B-Instruct"
model = "deep-learning-pytorch-huggingface/training/runs/llama-3.2-3b-r1-countdown/checkpoint-500"
model = "deep-learning-pytorch-huggingface/training/runs/llama-3.2-3b-r1-countdown/checkpoint-1000"
model = "deep-learning-pytorch-huggingface/training/runs/llama-3.2-3b-r1-countdown/checkpoint-1500"
model = "deep-learning-pytorch-huggingface/training/runs/llama-3.2-3b-r1-countdown/checkpoint-2000"


# gold_nums = [3, 2, 14, 71]
# gold_completion = """ To get an equation that equals 52 using the numbers 3, 2, 14, and 71 exactly once, I need to try different combinations of operations.\nLet's try: 71 - 14 - 2 - 3\n71 - 14 = 57\n57 - 2 = 55\n55 - 3 = 52\nThis works! </think>\n<answer> 71 - 14 - 2 - 3 </answer>"""

# at least 3 completions are correct:
# >>>completion=" To get an equation that equals 52 using the numbers 3, 2, 14, and 71 exactly once, I need to try different combinations of operations.\nLet's try: 71 - 14 - 2 - 3\n71 - 14 = 57\n57 - 2 = 55\n55 - 3 = 52\nThis works! </think>\n<answer> 71 - 14 - 2 - 3 = 52 </answer>"

# >>>completion=" To get an equation that equals 96 using the numbers 14, 77, 45, and 50 exactly once, I need to try different combinations of operations.\nLet's try: 77 + 45 - 50 + 14\n77 + 45 = 122\n122 - 50 = 72\n72 + 14 = 86 (not 96)\nLet's try: 77 + 50 - 45 + 14\n77 + 50 = 127\n127 - 45 = 82\n82 + 14 = 96 </think>\n<answer> (77 + 50 - 45 + 14) = 96 </answer>"

# >>>completion=" To get an equation that equals 21 using the numbers 43, 31, and 33 exactly once, I need to try different combinations of operations.\nLet's try: 43 - 31 - 33\n43 - 31 = 12\n12 - 33 = -21 (not 21)\nLet's try: 43 - 31 + 33\n43 - 31 = 12\n12 + 33 = 45 (not 21)\nLet's try: 43 + 31 - 33\n43 + 31 = 74\n74 - 33 = 41 (not 21)\nLet's try: 33 + 31 - 43\n33 + 31 = 64\n64 - 43 = 21 </think>\n<answer> 33 + 31 - 43 = 21 </answer>"


tokenizer = AutoTokenizer.from_pretrained(model)

# We'll set the pad token later, after the model is loaded


# llama 3.2 3b doesn't generate correct format (no <think> tags) or
# correct answer - https://colab.research.google.com/drive/1c97NDd-Ns6izPkprZnepHJmxeHkNCOmb#scrollTo=gBJN3muUCfo3


# generate r1 prompt with a prefix for the model to already start with the thinking process
def generate_r1_prompt(numbers, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
        },
        {
            "role": "user",
            "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.",
        },
        {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            r1_prefix, tokenize=False, continue_final_message=True
        ),
        "target": target,
    }


# convert our dataset to the r1 prompt
dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))

# split the dataset into train and test
train_test_split = dataset.train_test_split(test_size=0.1)

train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
print(len(train_dataset))  # 45000
print(len(test_dataset))  # 5000

# print(train_dataset[0])
# {'target': 52, 'nums': [96, 93, 91, 29], 'prompt': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 15 Jul 2025\n\nYou are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nUsing the numbers [96, 93, 91, 29], create an equation that equals 52. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nLet me solve this step by step.\n<think>'}
# print("*******")
# print(test_dataset[0])
# exit()


def format_reward_func(completions, target, **kwargs):
    """
    Format: <think>...</think><answer>...</answer>
    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers

      Returns:
          list[float]: Reward scores
    """
    rewards = []

    for completion, gt in zip(completions, target):

        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # Check if the format is correct
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

            match = re.search(regex, completion, re.DOTALL)
            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def equation_reward_func(completions, target, nums, **kwargs):
    """
    Evaluates completions based on:
    2. Mathematical correctness of the answer

    Args:
        completions (list[str]): Generated outputs
        target (list[str]): Expected answers
        nums (list[str]): Available numbers

    Returns:
        list[float]: Reward scores
    """
    rewards = []
    for completion, gt, numbers in zip(completions, target, nums):
        try:
            # add synthetic <think> as its already part of the prompt and prefilled for the assistant to more easily match the regex
            completion = "<think>" + completion
            # If completion contains "= .." (i.e., equals sign followed by two dots), replace it with ""
            completion = re.sub(r"=\s*\d+", "", completion)

            # completion = gold_completion
            # Check if the format is correct
            match = re.search(r"<answer>(.*?)<\/answer>", completion)
            if match is None:
                rewards.append(0.0)
                continue
            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # print(f"{equation=}")
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
            # print(f"{used_numbers=}")

            # numbers = gold_nums
            # gt = 52

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(numbers):
                # print(f">>>>{sorted(numbers)=} {sorted(used_numbers)=}")
                rewards.append(0.0)
                continue
            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r"^[\d+\-*/().=\s]+$"
            if not re.match(allowed_pattern, equation):
                # print(f"{equation=} not allowed")
                rewards.append(0.0)
                continue

            # print(f"{equation=} allowed")

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # print(f"{result=}")

            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(gt)) < 1e-5:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception as e:
            # If evaluation fails, reward is 0
            rewards.append(0.0)
            print(e)
    return rewards


correct_sample_1 = """We need to find an equation using the numbers 19, 36, 55, and 7
exactly once, with basic arithmetic operations, that equals 65. One possible
combination is 55 + 36 - 19 + 7... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""

correct_sample_2 = """ ... </think>
<answer> 55 + 36 - 7 - 19 </answer>"""

wrong_format = (
    """User: Using the numbers [19, 36, 55, 7], create an equation that equals 65."""
)

wrong_format_2 = """To find the equation that equals 79 using the numbers 95, 78, 6, 88, I'll start by adding 88 and 95:
95 + 88 = 183
Now, let's subtract 104 from 183 to get 79:
183 - 104 = 79
<think> 183 - 104 = 79 </think><think> 183 - 104 = 79 </think><answer> 183 - 104 = 79 </answer>"""

wrong_result = """ ... </think>
<answer> 55 + 36 - 7 - 18 </answer>"""


test_rewards = format_reward_func(
    completions=[
        correct_sample_1,
        correct_sample_2,
        wrong_format,
        wrong_format_2,
        wrong_result,
    ],
    target=["65", "65", "65", "65", "65"],
    nums=[[19, 36, 55, 7]] * 5,
)
assert test_rewards == [1.0, 1.0, 0.0, 0.0, 1.0], "Reward function is not working"
# test_rewards = equation_reward_func(
#     completions=[
#         correct_sample_1,
#         correct_sample_2,
#         wrong_format,
#         wrong_format_2,
#         wrong_result,
#     ],
#     target=["65", "65", "65", "65", "65"],
#     nums=[[19, 36, 55, 7]] * 5,
# )
# assert test_rewards == [1.0, 1.0, 0.0, 0.0, 0.0], "Reward function is not working"

# our model we are going to use as policy
model_config = ModelConfig(
    model_name_or_path=model,  # "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="bfloat16",
    attn_implementation="flash_attention_2",
    use_peft=True,
    load_in_4bit=True,
)

# Hyperparameters - optimized for lower memory usage
training_args = GRPOConfig(
    output_dir="llm_r1_outputs",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=10,
    max_steps=100,  # 250,  # num_generations=4: 2/450 [53:54<201:10:52
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,  # Increased from 1 to 4 to reduce memory usage
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific parameters
    max_prompt_length=256,  # Reduced from 256 to 128
    max_completion_length=1024,  # Reduced from 1024 to 512
    num_generations=1,  # 2,  # Reduced from 2 to 1
    beta=0.001,
    # Additional memory optimization
    # optim="adamw_8bit",  # Use 8-bit Adam optimizer adamw_8bit to save memory
    # max_grad_norm=1.0,  # Clip gradients to prevent memory spikes
)
# Set up the tokenizer with a pad token before initializing the trainer
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=get_peft_config(model_config),
)

# Make sure the trainer uses our tokenizer with the pad token set
trainer.tokenizer = tokenizer
# 3 7 9 13
# 2, 3, 5, 12
# 2, 5, 5, 10

# Initialize OpenAI client for single vLLM server with tensor parallelism
# Since you're running: CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve <model>
# This creates ONE server with tensor parallelism across 4 GPUs
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Single vLLM server URL
    api_key="dummy",  # vLLM doesn't require a real API key
)


def generate_completion_api(prompt, max_tokens=512, client_idx=0):
    """
    Generate completion using vLLM OpenAI API (single server with tensor parallelism)
    """
    try:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=0.9,
        )
        return response.choices[0].text
    except Exception as e:
        print(f"API call failed: {e}")
        return ""


def evaluate_with_api(test_dataset, num_samples=5000, max_workers=8):
    """
    Evaluate the model's performance using vLLM API with parallel processing

    Args:
        test_dataset: The test dataset to evaluate on
        num_samples: Number of samples to evaluate (default: 5000)
        max_workers: Number of parallel workers (default: 8)

    Returns:
        dict: Performance metrics
    """
    print(f"Evaluating model on {num_samples} test samples using vLLM API...")

    # Select a subset of test samples for evaluation
    eval_samples = test_dataset.shuffle(seed=42).select(
        range(min(num_samples, len(test_dataset)))
    )

    # Generate completions for the test samples
    prompts = [sample["prompt"] for sample in eval_samples]
    targets = [sample["target"] for sample in eval_samples]
    nums = [sample["nums"] for sample in eval_samples]

    # Generate completions using parallel API calls
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all API calls with load balancing across servers
        futures = [
            executor.submit(generate_completion_api, prompt, 512, i)
            for i, prompt in enumerate(prompts)
        ]

        # Collect results with progress bar
        completions = []
        for future in tqdm(futures, desc="Generating completions"):
            completions.append(future.result())

    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")
    print(
        f"Average time per sample: {(end_time - start_time) / len(prompts):.3f} seconds"
    )

    # Evaluate using reward functions
    format_rewards = format_reward_func(completions, targets)
    equation_rewards = equation_reward_func(completions, targets, nums)

    # Calculate metrics
    format_accuracy = sum(format_rewards) / len(format_rewards)
    equation_accuracy = sum(equation_rewards) / len(equation_rewards)

    metrics = {
        "format_accuracy": format_accuracy,
        "equation_accuracy": equation_accuracy,
        "num_samples": len(eval_samples),
        "generation_time": end_time - start_time,
        "avg_time_per_sample": (end_time - start_time) / len(prompts),
    }

    print(f"Format Accuracy: {format_accuracy:.4f}")
    print(f"Equation Accuracy: {equation_accuracy:.4f}")
    print(f"Samples evaluated: {len(eval_samples)}")

    return metrics


def evaluate(trainer, test_dataset, num_samples=5000):
    """
    Evaluate the model's performance on the test dataset

    Args:
        trainer: The GRPOTrainer instance
        test_dataset: The test dataset to evaluate on
        num_samples: Number of samples to evaluate (default: 100)

    Returns:
        dict: Performance metrics
    """
    print(f"Evaluating model on {num_samples} test samples...")

    # Select a subset of test samples for evaluation
    eval_samples = test_dataset.shuffle(seed=42).select(
        range(min(num_samples, len(test_dataset)))
    )

    # Generate completions for the test samples
    prompts = [sample["prompt"] for sample in eval_samples]
    targets = [sample["target"] for sample in eval_samples]
    nums = [sample["nums"] for sample in eval_samples]

    # Generate completions using the trainer's model
    completions = []
    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        # Generate completion
        # print(f">>>{prompt=}")
        inputs = trainer.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(trainer.model.device) for k, v in inputs.items()}

        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                # temperature=0.0,
                pad_token_id=trainer.tokenizer.eos_token_id,
            )

        # Decode the completion (remove the prompt part)
        completion = trainer.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        )
        # print(f">>>{completion=}")
        completions.append(completion)

    # Evaluate using reward functions
    format_rewards = format_reward_func(completions, targets)
    equation_rewards = equation_reward_func(completions, targets, nums)
    # print(f">>>{format_rewards=}")
    # print(f">>>{equation_rewards=}")

    # Calculate metrics
    format_accuracy = sum(format_rewards) / len(format_rewards)
    equation_accuracy = sum(equation_rewards) / len(equation_rewards)
    # print(f">>>{format_accuracy=}")
    # print(f">>>{equation_accuracy=}")

    metrics = {
        "format_accuracy": format_accuracy,
        "equation_accuracy": equation_accuracy,
        "num_samples": len(eval_samples),
    }

    print(f"Format Accuracy: {format_accuracy:.4f}")
    print(f"Equation Accuracy: {equation_accuracy:.4f}")
    print(f"Samples evaluated: {len(eval_samples)}")

    return metrics


# Evaluate model performance using vLLM API
print("=" * 50)
print(f"EVALUATING MODEL WITH vLLM API for {model}")
print("=" * 50)

# Use the new API-based evaluation (tensor parallelism across 4 GPUs)
# Since you're running: CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve <model>
# This is ONE server with tensor parallelism - use fewer concurrent requests
# - Using max_workers=8 for optimal performance with tensor parallelism
# - All 4 GPUs work together on each request
pre_training_metrics = evaluate_with_api(test_dataset, num_samples=5000, max_workers=8)


# Run
# python eval_llm_grpo_trained_vllm.py
# after running something like:
# CUDA_VISIBLE_DEVICES=2,3,4,5 vllm serve <model_name_or_path> --tensor-parallel-size 4
# where <model_name_or_path> is meta-llama/Llama-3.2-3B-Instruct or the path to the model checkpoint
