"""Qwen3 non-thinking generation and batched DART-Math answer judging."""

import contextlib
import hashlib
import json
import random
from types import SimpleNamespace

from stage_one.model_runner import install_execution_cache


class ShowcaseModelRunner:
    def __init__(self, args, log_stream):
        import torch
        from dart_math.eval import EvaluatorMathBatch
        from llm_depth_router.model import get_model, get_tokenizer

        torch.cuda.set_device(args.device)
        self.model = get_model(args.model_path, device=f"cuda:{args.device}")
        self.model.eval()
        self.tokenizer = get_tokenizer(args.model_path)
        self.hook = install_execution_cache(self.model)
        self.evaluator = EvaluatorMathBatch(
            strict_extract=True, use_orig_eq_for_olympiadbench=True, timeout=60
        )
        self.args = args
        self.log_stream = log_stream
        self.depth = int(self.model.config.num_hidden_layers)
        if getattr(self.model.config, "quantization_config", None):
            raise ValueError("Use the configured full, unquantized Qwen3-8B snapshot")

    def close(self):
        self.hook.remove()

    def metadata(self):
        return {"model_class": type(self.model).__name__, "depth": self.depth,
                "dtype": str(self.model.dtype), "training": self.model.training,
                "attention": self.model.config._attn_implementation,
                "thinking": False, "temperature": self.args.temperature,
                "max_new_tokens": self.args.max_new_tokens,
                "cache_adapter": "execution-slot DynamicCache"}

    def _seed(self, sample_id, path):
        import numpy as np
        import torch

        token = json.dumps([self.args.seed, sample_id, path], separators=(",", ":"))
        seed = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def generate(self, question, sample_id, path):
        import torch
        from llm_depth_router.model import setup_custom_path
        from polar.eval import _qwen3_apply_chat_template, _qwen3_split_thinking

        self._seed(sample_id, path)
        prompt = (
            "Solve the following math problem and output ONLY the final answer directly, "
            "formatted strictly as \\boxed{ANSWER}.\n"
            "### Problem Start\n"
            f"{question}\n"
            "### Problem End\n"
            "Answer:"
        )
        text = _qwen3_apply_chat_template(self.tokenizer, prompt)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        setup_custom_path(self.model, path)
        generation = {"max_new_tokens": self.args.max_new_tokens,
                      "do_sample": self.args.temperature > 0}
        if self.args.temperature > 0:
            generation["temperature"] = self.args.temperature
        with torch.inference_mode(), contextlib.redirect_stdout(self.log_stream), \
                contextlib.redirect_stderr(self.log_stream):
            output = self.model.generate(**inputs, **generation)
        output_ids = output[0, inputs.input_ids.shape[1]:].tolist()
        thinking, answer = _qwen3_split_thinking(output_ids, self.tokenizer)
        if thinking:
            raise RuntimeError("Qwen3 emitted a thinking trace despite enable_thinking=False")
        return answer.strip()

    def judge_batch(self, generated_texts, ground_truth):
        """Judge one search batch in one multiprocessing call, not once per path."""
        valid_indices = [index for index, text in enumerate(generated_texts)
                         if "oxed{" in text]
        results = [("", False) for _ in generated_texts]
        if not valid_indices:
            return results
        samples = [SimpleNamespace(resp=generated_texts[index], ref_ans=ground_truth,
                                   ans=None, query="", dataset="math")
                   for index in valid_indices]
        with contextlib.redirect_stdout(self.log_stream), contextlib.redirect_stderr(self.log_stream):
            extracted, correct = self.evaluator.batch_eval(samples, n_procs=4)
        for index, answer, score in zip(valid_indices, extracted, correct):
            results[index] = ("" if answer is None else str(answer), bool(score))
        return results
