"""Qwen3 non-thinking generation and bounded DART-Math answer judging."""

import contextlib
import hashlib
import json
import random
import signal
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
        # Match the released evaluator: score content after </think> instead of
        # failing the whole search. Altered layer paths can emit unexpected
        # control tokens even when the normal chat template disables thinking.
        return answer.strip(), bool(thinking)

    def judge(self, generated_text, ground_truth):
        """Run the same DART-Math extractor/equivalence logic with a hard timeout."""
        if "oxed{" not in generated_text:
            return "", False

        class JudgingTimeout(TimeoutError):
            pass

        def timeout_handler(signum, frame):
            del signum, frame
            raise JudgingTimeout("DART-Math judging exceeded 60 seconds")

        previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, 60)
        try:
            with contextlib.redirect_stdout(self.log_stream), \
                    contextlib.redirect_stderr(self.log_stream):
                extracted = self.evaluator.extract_ans(generated_text)
                sample = SimpleNamespace(resp=generated_text, ref_ans=ground_truth,
                                         ans=extracted, query="", dataset="math")
                correct = self.evaluator.eval(sample)
            return "" if extracted is None else str(extracted), bool(correct)
        except Exception as exc:
            print(f"Judge failure treated as incorrect: {type(exc).__name__}: {exc}",
                  file=self.log_stream)
            return "", False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
