import re
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


format_pattern = r"^<think>(?:(?!</think>).)*</think><answer>(?:(?!</answer>).)*</answer>\Z"


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
        # We need to modify code here to print debug information if verification fails
        # https://github.com/huggingface/Math-Verify/blob/6f5218c822fb2ed2d3618d6eda501295069b4061/src/math_verify/grader.py#L836
        reward = float(verify(answer_parsed, gold_parsed))
  
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
    if not ground_truth.startswith("$"):
        ground_truth = "$" + ground_truth + "$"
    
    if not solution_str:
        return 0.0
    
    format_reward = 0.0
    if enable_format_reward:
        format_reward = float(verify_format(solution_str)) * 0.5
    acc_reward = verify_math(solution_str, ground_truth)

    reward = format_reward + acc_reward

    return {
        "score": reward,
        "acc_reward": acc_reward,
        "format_reward": format_reward,
    }