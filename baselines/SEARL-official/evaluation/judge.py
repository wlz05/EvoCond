# -*- coding: utf-8 -*-
# EvoCond: Preserving Conditional Tool Relations for Co-Evolving Language Agents

import os
from typing import Optional

from vllm import LLM, SamplingParams

'\n\u57fa\u4e8e vLLM \u7684\u5f00\u6e90\u6a21\u578b\u5224\u5206\u5668\uff0c\u9ed8\u8ba4\u4f7f\u7528\u5f00\u6e90\u6743\u91cd\uff08\u53ef\u901a\u8fc7\u73af\u5883\u53d8\u91cf/\u5165\u53c2\u66ff\u6362\uff09\u3002\n\u73af\u5883\u53d8\u91cf\uff1a\n- VLLM_MODEL_PATH\uff1a\u6a21\u578b\u6743\u91cd\uff08HF \u540d\u79f0\u6216\u672c\u5730\u8def\u5f84\uff09\uff0c\u9ed8\u8ba4 Qwen/Qwen2.5-7B-Instruct\n- VLLM_TP_SIZE\uff1atensor parallel \u5927\u5c0f\uff0c\u9ed8\u8ba4 1\n- VLLM_GPU_MEM_UTIL\uff1aGPU \u663e\u5b58\u5229\u7528\u7387\uff0c\u9ed8\u8ba4 0.9\n'

DEFAULT_MODEL = os.environ.get("VLLM_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

class VLLMJudge:
    '\u4f7f\u7528 vLLM \u90e8\u7f72\u7684\u5f00\u6e90\u6a21\u578b\u505a\u6570\u503c\u7b54\u6848\u5224\u5206\u3002'

    _shared_llm: Optional[LLM] = None
    _shared_model_path: Optional[str] = None

    def __init__(
        self,
        model_path: Optional[str] = None,
        temperature: float = 0.0,
        max_new_tokens: int = 8,
        batch_size: int = 256,
    ):
        self.model_path = model_path or DEFAULT_MODEL
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._ensure_llm()

    def _ensure_llm(self):
        if (VLLMJudge._shared_llm is None) or (
            VLLMJudge._shared_model_path != self.model_path
        ):
            tp = int(os.environ.get("VLLM_TP_SIZE", "1"))
            gpu_util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.9"))
            VLLMJudge._shared_model_path = self.model_path
            VLLMJudge._shared_llm = LLM(
                model=self.model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_util,
            )

    @property
    def llm(self) -> LLM:
        return VLLMJudge._shared_llm

    def _build_prompt(self, prediction: str, reference: str, question: str) -> str:
        content = (
            f"question: {question}\n"
            f"prediction: {prediction}\n"
            f"reference: {reference}\n"
        )
        return [
            {"role": "system", "content": "You are a strict answer grader. Compare the prediction with the reference answer to determine if they convey the same information or answer. Consider:\n1. For numerical answers: check if they are mathematically equivalent (e.g., 0.5 = 1/2 = 50%)\n2. For text answers: check if the core meaning matches, ignoring minor formatting differences\n3. Ignore case, whitespace, and punctuation differences\n\nReturn ONLY:\n- '1' if prediction matches reference\n- '0' if prediction does NOT match reference\n\nOutput only a single digit: 1 or 0, nothing else."},
            {"role": "user", "content": content}
        ]

    def _call_llm(self, messages: list) -> str:
        '\u5355\u6837\u672c\u63a8\u7406\uff08\u517c\u5bb9\u65e7\u63a5\u53e3\uff09'
        results = self._call_llm_batch([messages])
        return results[0] if results else ""

    def _call_llm_batch(self, messages_list: list) -> list:
        '\u6279\u6b21\u63a8\u7406'
        if not messages_list:
            return []

        tokenizer = self.llm.get_tokenizer()

        prompts = []
        for messages in messages_list:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            prompts.append(prompt)

        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            stop=["\n"]
        )

        outputs = self.llm.generate(prompts, params)

        results = []
        for output in outputs:
            if output.outputs:
                results.append(output.outputs[0].text.strip())
            else:
                results.append("")

        return results

    def judge(self, prediction: str, reference: str, question: str) -> float:
        '\n        \u8fd4\u56de 1.0 / 0.0 \u8868\u793a\u662f\u5426\u5339\u914d\u3002\n        '
        results = self.judge_batch([(prediction, reference, question)])
        return results[0] if results else 0.0

    def judge_batch(self, items: list) -> list:
        '\n        \u6279\u6b21\u5224\u65ad\u591a\u4e2a\u6837\u672c\n        items: list of (prediction, reference, question) tuples\n        \u8fd4\u56de: list of float scores\n        '
        if not items:
            return []

        messages_list = []
        for prediction, reference, question in items:
            prompt = self._build_prompt(prediction, reference, question)
            messages_list.append(prompt)

        all_results = []
        for i in range(0, len(messages_list), self.batch_size):
            batch_messages = messages_list[i:i + self.batch_size]
            try:
                batch_responses = self._call_llm_batch(batch_messages)
                batch_scores = []
                for resp in batch_responses:

                    resp_clean = resp.strip()
                    if resp_clean == "1":
                        batch_scores.append(1.0)
                    elif resp_clean == "0":
                        batch_scores.append(0.0)
                    else:

                        if "1" in resp_clean and "0" not in resp_clean:
                            print(f"[VLLMJudge] Warning: non-standard response '{resp}', extracted as 1")
                            batch_scores.append(1.0)
                        elif "0" in resp_clean and "1" not in resp_clean:
                            print(f"[VLLMJudge] Warning: non-standard response '{resp}', extracted as 0")
                            batch_scores.append(0.0)
                        else:

                            print(f"[VLLMJudge] Error: unparseable response '{resp}', defaulting to 0")
                            batch_scores.append(0.0)
                all_results.extend(batch_scores)
            except Exception as e:
                print(f"[VLLMJudge] batch inference failed: {e}")

                all_results.extend([0.0] * len(batch_messages))

        return all_results

    def evaluation(self, res: str, golden: str, question: str) -> float:
        '\n        \u517c\u5bb9\u65e7\u63a5\u53e3\uff0c\u76f4\u63a5\u8c03\u7528 judge\u3002\n        '
        return self.judge(res, golden, question)

    def evaluation_batch(self, items: list) -> list:
        '\n        \u6279\u6b21\u8bc4\u4f30\u591a\u4e2a\u6837\u672c\n        items: list of (res, golden, question) tuples\n        \u8fd4\u56de: list of float scores\n        '
        return self.judge_batch(items)

MathLLM = VLLMJudge

def main_debug(prompt: str, reference: str, question: str):
    judge = VLLMJudge()
    score = judge.judge(prompt, reference, question)
    print("judge score:", score)

if __name__ == "__main__":
    demo_pred = "The answer is 42."
    demo_ref = "42"
    demo_q = "What is 6 * 7?"
    main_debug(demo_pred, demo_ref, demo_q)

# Author: Weiliang Zhu
# Email: wlzchina05@gmail.com
