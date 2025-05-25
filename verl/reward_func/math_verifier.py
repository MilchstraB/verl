import os
import re
import random
import asyncio
import uvicorn
from argparse import ArgumentParser

from fastapi import FastAPI, Request, HTTPException
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from loguru import logger

app = FastAPI()


def get_response_from_query(q: str):
    ends_of_sentence = ["<|im_end|>", "<｜end▁of▁sentence｜>", "<|endoftext|>", "<end_of_turn>"]
    pos = re.search(response_prefix, q)
    if pos is None:
        return None
    response = q[pos.end() :]
    for e in ends_of_sentence:
        response = response.replace(e, "")
    return response.strip()


def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1


def verify_math(content, sol):
    gold_parsed = parse(
        sol,
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig()],
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
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as e:
            reward = 0.0
            print("Failed to verify: ", e)
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", sol)
    return reward


@app.post("/get_reward")
async def get_reward(request: Request):
    json_data = await request.json()

    def wrap_process_data():
        required_fields = ["response", "prompt", "ground_truth"]
        missing_fields = [field for field in required_fields if field not in json_data]
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Missing required fields: {', '.join(missing_fields)}")
        
        prompt = json_data["prompt"]
        q = json_data["response"]
        ground_truth = json_data["ground_truth"]

        if prompt is None:
            raise HTTPException(status_code=400, detail=f"Problem not found from {q}")
        if not ground_truth.startswith("$"):
            ground_truth = "$" + ground_truth + "$"

        response = get_response_from_query(q) or q
        if response is None:
            raise HTTPException(status_code=400, detail=f"Response not found from {q}")
        
        # Apply format reward only if enabled
        format_reward = 0.0
        if enable_format_reward:
            format_reward = float(verify_format(response)) * 0.5
        acc_reward = verify_math(response, ground_truth)

        do_print = random.randint(1, 20) == 1
        if do_print:
            info = f"Query: {q}\n\nProblem: {prompt}\n\n Answer: {ground_truth}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward}\n\n"
            info = re.sub(r"<\|.*?\|>|<pad>", "", info)
            logger.info(info)

        return {
            "score": format_reward + acc_reward,
            "reward_extra_info": {
                "format_reward": format_reward,
                "acc_reward": acc_reward
            }
        }

    result = await asyncio.to_thread(wrap_process_data)
    return result        


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompt-template", type=str, default=None, help="Prompt template", required=True)
    parser.add_argument("--log_file", type=str, default="remote_rm.log", help="Log file path")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    parser.add_argument(
        "--disable-format-reward",
        action="store_true",
        help="Disable format reward calculation. When enabled (default), responses get +0.5 reward for correct format.",
    )
    args = parser.parse_args()
    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    logger.remove()
    logger.add(args.log_file)

    # Set format reward flag based on command line argument
    enable_format_reward = not args.disable_format_reward
    print(f"Format reward is {'disabled' if args.disable_format_reward else 'enabled'}")
    logger.info(f"Format reward is {'disabled' if args.disable_format_reward else 'enabled'}")

    logger.warning(
        "Math-verify is thread-unsafe! Please refer to "
        "https://github.com/huggingface/Math-Verify/issues/50#issuecomment-2835692251 for more details."
    )

    format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"

    if args.prompt_template == "chatml":
        response_prefix = r"<\|im_start\|>assistant\n"
    elif args.prompt_template == "qwen1":
        response_prefix = r"<｜Assistant｜>"
    elif args.prompt_template == "base":
        response_prefix = r"Assistant: "
    elif args.prompt_template == "phi3":
        response_prefix = r"<|assistant|>\n"
    elif args.prompt_template == "phi4":
        response_prefix = r"<|assistant|>\n"
    elif args.prompt_template == "gemma3":
        response_prefix = r"<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown chat format: {args.prompt_template}")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
