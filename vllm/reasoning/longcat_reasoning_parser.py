# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser


class LongcatReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for LongCat models that emit <longcat_think> tokens.

    If the model output does not contain LongCat thinking tags, treat the entire
    output as content.
    """

    @property
    def start_token(self) -> str:
        return "<longcat_think>"

    @property
    def end_token(self) -> str:
        return "</longcat_think>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        if self.start_token_id not in input_ids and self.end_token_id not in input_ids:
            return True
        return super().is_reasoning_end(input_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Sequence[int]
    ) -> bool:
        if self.start_token_id not in input_ids and self.end_token_id not in input_ids:
            return True
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.start_token_id not in input_ids and self.end_token_id not in input_ids:
            return input_ids
        return super().extract_content_ids(input_ids)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)
