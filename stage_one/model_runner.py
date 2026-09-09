"""Reuse official generation and scoring; isolate KV state per execution slot."""

import contextlib
import hashlib
import json
import random

from .storage import file_digest, relative_path


def model_inventory(model_path):
    """Fingerprint a local snapshot without loading any weight tensors."""
    path = relative_path(model_path)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"Prepare a complete local model snapshot at {path}")
    files = []
    from tqdm import tqdm
    for item in tqdm(sorted(path.rglob("*")), desc="Fingerprint model files", leave=False):
        if item.is_file() and ".cache" not in item.parts:
            row = {"file": str(item.relative_to(path)), "bytes": item.stat().st_size,
                   "sha256": file_digest(item)}
            files.append(row)
    if not any(row["file"].endswith((".safetensors", ".bin")) for row in files):
        raise ValueError(f"No model weight files at {path}")
    if not (path / "tokenizer_config.json").is_file() or not any(
            (path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ValueError(f"Missing tokenizer assets at {path}")
    for item in path.rglob("*"):
        if item.is_file() and item.suffix in {".safetensors", ".bin"}:
            with item.open("rb") as stream:
                if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                    raise ValueError(f"Weight is a Git LFS pointer, not downloaded model bytes: {item}")
    # Catch incomplete sharded snapshots before GPU allocation.
    for index in path.glob("*.index.json"):
        data = json.loads(index.read_text(encoding="utf-8"))
        for filename in set(data.get("weight_map", {}).values()):
            if not (path / relative_path(filename)).is_file():
                raise ValueError(f"Missing model shard {filename}")
    return files


def install_execution_cache(model):
    """Keep original layer IDs and attention behavior; remap only cache slots.

    HF 4.52.4 attention calls cache.update(..., self.layer_idx, ...). Reusing
    that slot during a loop appends the same token twice to a shared cache.
    This instance hook supplies a DynamicCache whose slots follow custom_path.
    No weights, original attention IDs, or generation options are modified.
    """
    from transformers.cache_utils import DynamicCache

    class ExecutionCache(DynamicCache):
        def begin(self, path):
            path = tuple(path)
            previous = getattr(self, "execution_path", None)
            if previous is not None:
                if previous != path or self.execution_slot != len(path):
                    raise RuntimeError("Path changed or prior forward did not execute the complete program")
            self.execution_path = path
            self.execution_slot = 0

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            slot = self.execution_slot
            if slot >= len(self.execution_path) or self.execution_path[slot] != layer_idx:
                raise RuntimeError("Observed attention execution differs from custom_path")
            self.execution_slot += 1
            return super().update(key_states, value_states, slot, cache_kwargs)

    def before_forward(backbone, args, kwargs):
        if not kwargs.get("use_cache", backbone.config.use_cache):
            return args, kwargs
        cache = kwargs.get("past_key_values")
        if not isinstance(cache, ExecutionCache):
            if cache is not None and (type(cache) is not DynamicCache or cache.get_seq_length() != 0):
                raise RuntimeError("Only fresh DynamicCache is supported for a new execution program")
            cache = ExecutionCache()
        cache.begin(backbone.custom_path)
        kwargs["past_key_values"] = cache
        return args, kwargs

    return model.model.register_forward_pre_hook(before_forward, with_kwargs=True)


class ModelRunner:
    def __init__(self, args, local_rank, log_stream):
        import torch
        from llm_depth_router.model import get_model, get_tokenizer
        import polar.eval as polar_eval

        torch.cuda.set_device(local_rank)
        self.model = get_model(args.model_path, device=f"cuda:{local_rank}")
        self.tokenizer = get_tokenizer(args.model_path)
        self.hook = install_execution_cache(self.model)
        self.evaluate = polar_eval._online_eval_math_single
        self.polar_eval = polar_eval
        self.args = args
        self.log_stream = log_stream
        self.depth = int(self.model.config.num_hidden_layers)
        if getattr(self.model.config, "quantization_config", None):
            raise ValueError("A full, unquantized base model is required")

    def metadata(self):
        return {"class": type(self.model).__name__, "dtype": str(self.model.dtype),
                "training": self.model.training, "depth": self.depth,
                "attention": self.model.config._attn_implementation,
                "generation_config": self.model.generation_config.to_dict(),
                "cache_adapter": "execution-slot DynamicCache"}

    def score(self, row, path):
        import numpy as np
        import torch
        # Same question/path yields the same RNG seed across rank counts/resumes.
        token = json.dumps([self.args.seed, row["sample_id"], path], separators=(",", ":"))
        seed = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # Official evaluator progress is captured in the rank log. Outer progress
        # bars remain visible, avoiding thousands of overlapping 1-item bars.
        with contextlib.redirect_stdout(self.log_stream), contextlib.redirect_stderr(self.log_stream):
            return self.evaluate(model=self.model, tokenizer=self.tokenizer,
                                 model_path=self.args.model_path, transition=path,
                                 question=row["question"], gt=row["gt_ans"],
                                 max_new_tokens=self.args.max_new_tokens,
                                 temperature=self.args.temperature)

    def residual_stream(self, row, path, pooling):
        """Return post-block prefill residuals in execution-slot order."""
        import torch
        from llm_depth_router.model import setup_custom_path

        prompt = (
            "Solve the following math problem and output ONLY the final answer directly, "
            "formatted strictly as \\boxed{ANSWER}.\n"
            "### Problem Start\n"
            f"{row['question']}\n"
            "### Problem End\n"
            "Answer:"
        )
        helper = self.polar_eval
        if helper._is_qwen3_model_path(self.args.model_path):
            text = helper._qwen3_apply_chat_template(self.tokenizer, prompt)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        elif helper._is_qwen15_moe_chat_model_path(self.args.model_path):
            text = helper._qwen15_moe_apply_chat_template(self.tokenizer, prompt)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        elif helper._is_qwen25_instruct_model_path(self.args.model_path):
            text = helper._qwen25_apply_chat_template(self.tokenizer, prompt)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        setup_custom_path(self.model, path)
        captured = []
        mask = inputs.get("attention_mask")

        def capture_residual(module, layer_inputs, layer_output):
            del module, layer_inputs
            state = layer_output[0] if isinstance(layer_output, tuple) else layer_output
            if pooling == "last-token":
                vector = state[0, -1]
            elif pooling == "mean":
                if mask is None:
                    vector = state[0].mean(dim=0)
                else:
                    weights = mask[0].to(dtype=state.dtype).unsqueeze(-1)
                    vector = (state[0] * weights).sum(dim=0) / weights.sum().clamp_min(1)
            else:
                raise ValueError(f"Unknown residual pooling: {pooling}")
            # Clone a single vector so last-token pooling does not retain the
            # full sequence tensor. Transfer all slots to CPU together below.
            captured.append(vector.detach().clone())

        # A repeated layer module invokes the same hook repeatedly, so hook
        # order is exactly the custom program's execution-slot order. Capturing
        # at decoder-block outputs also avoids mixing a final normalized hidden
        # state with intermediate pre-normalization residuals.
        handles = [self.model.model.layers[index].register_forward_hook(capture_residual)
                   for index in sorted(set(path))]
        try:
            with torch.no_grad():
                self.model(**inputs, use_cache=False, output_hidden_states=False,
                           return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(path):
            raise RuntimeError("Residual stream length does not match execution path")
        matrix = torch.stack(captured).float().cpu().numpy()
        return matrix, int(inputs["input_ids"].shape[1])
