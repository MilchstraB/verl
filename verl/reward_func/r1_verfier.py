import re
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
except ImportError:
    print("To use R1-Verify, please install it first by running `pip install math-verify latex2sympy2_extended`.")


format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"


def get_response_from_query(q: str, response_prefix: str):
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


def compute_score(
    solution_str: str,
    ground_truth: str,
    prompt_template: str = "chatml",
    enable_format_reward: bool = True,
    **kwargs,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        prompt_template: The prompt template to use
        enable_format_reward: Whether to use format reward

    Returns:
        Reward score (1.0 for correct, 0.0 for incorrect)
    """
    if prompt_template == "chatml":
        response_prefix = r"<\|im_start\|>assistant\n"
    elif prompt_template == "qwen1":
        response_prefix = r"<｜Assistant｜>"
    elif prompt_template == "base":
        response_prefix = r"Assistant: "
    elif prompt_template == "phi3":
        response_prefix = r"<|assistant|>\n"
    elif prompt_template == "phi4":
        response_prefix = r"<|assistant|>\n"
    elif prompt_template == "gemma3":
        response_prefix = r"<start_of_turn>model\n"
    else:
        raise ValueError(f"Unknown chat format: {prompt_template}")
    
    if not ground_truth.startswith("$"):
        ground_truth = "$" + ground_truth + "$"
    
    solution_str = get_response_from_query(solution_str, response_prefix) or solution_str
    if not solution_str:
        return 0.0
    
    format_reward = 0.0
    if enable_format_reward:
        format_reward = float(verify_format(solution_str)) * 0.5
    reward = verify_math(solution_str, ground_truth)

    # do_print = random.randint(1, 20) == 1
    # if do_print:
    #     info = f"Query: {q}\n\nProblem: {problem}\n\n Answer: {answer}\n\n Response: {response}\n\n Format Reward: {format_reward}\n\n Acc Reward: {acc_reward_future.result()}\n\n"
    #     info = re.sub(r"<\|.*?\|>|<pad>", "", info)
    #     logger.info(info)

    reward = format_reward + reward

    return {
        "score": reward,
        "format_reward": format_reward,
        "acc": reward == 1.0,
        "pred": solution_str,
    }