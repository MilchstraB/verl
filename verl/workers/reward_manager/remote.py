# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import aiohttp
from typing import List
from dataclasses import dataclass, field

import torch

from verl import DataProto


@dataclass
class CallScoreFuncInput:
    """The input to compute_score function."""
    response: str = field(default=None)
    prompt: str = field(default=None)
    ground_truth: str = field(default=None)
    extra_info: dict = field(default_factory=dict)
    data_source: str = field(default=None)
    valid_response_length: int = field(default=None)


@dataclass
class CallScoreFuncOutput:
    """The output of compute_score function."""
    score: float = field(default=None)
    reward_extra_info: dict = field(default_factory=dict)


async def async_call_online_reward_model(url: str, **kwargs):
    try:
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
        }

        json_data = {**kwargs}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=json_data) as response:
                res = await response.json()
                final_score = res.get("score")
                return float(final_score), res
    except Exception as e:
        return 0.0, {}


def call_online_reward_model(url: str, **kwargs):
    try:
        # Use the async function in a synchronous context
        loop = asyncio.get_event_loop()
        score, details = loop.run_until_complete(
            async_call_online_reward_model(url, **kwargs)
        )
        return score, details
    except Exception as e:
        return 0.0, {}


class RemoteRewardManager:
    """The reward manager."""

    def __init__(
            self,
            reward_api, 
            tokenizer, 
            num_examine, 
            compute_score=None,
            max_concurrency=30, 
            reward_fn_key="data_source"
        ) -> None:
        self.reward_api = reward_api
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.max_concurrency = max_concurrency
        self.compute_score = compute_score or call_online_reward_model
        self.reward_fn_key = reward_fn_key

    async def batch_compute_scores(
        self, input_list: List[CallScoreFuncInput], max_concurrency: int = 30
    ) -> List[CallScoreFuncOutput]:
        semaphore = asyncio.Semaphore(max_concurrency)

        async def bounded_compute_score(input_item: CallScoreFuncInput):
            async with semaphore:
                return await self.async_compute_score(input_item)

        tasks = [bounded_compute_score(input_item) for input_item in input_list]
        return await asyncio.gather(*tasks)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}
        scorefuncinput_list: List[CallScoreFuncInput] = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            scorefuncinput_list.append(
                CallScoreFuncInput(
                    response=response_str,
                    prompt=prompt_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    data_source=data_source,
                    valid_response_length=valid_response_length,
                )
            )


        score_funcoutput_list = asyncio.run(
            self.batch_compute_scores(
                scorefuncinput_list,
                self.max_concurrency,
            )
        )

        for i in range(len(data)):
            valid_response_length = scorefuncinput_list[i].valid_response_length
            score = score_funcoutput_list[i].score
            reward_extra_info = score_funcoutput_list[i].reward_extra_info
            reward_tensor[i, valid_response_length - 1] = score

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor, reward_extra_info
