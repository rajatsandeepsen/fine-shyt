from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from statistics import mean
from time import perf_counter
from typing import Any, Sequence, TypedDict

import torch
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/SmolLM2-360M-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastModel.for_inference(model)


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(slots=True)
class InferenceStats:
    prompt_index: int
    run_index: int
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_per_s: float
    total_tokens_per_s: float
    gpu_peak_mem_mb: float | None


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_once(
    *,
    model: Any,
    tokenizer: Any,
    messages: Sequence[ChatMessage],
    prompt_index: int,
    run_index: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> InferenceStats:
    device = next(model.parameters()).device
    rendered_prompt = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered_prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _sync_if_cuda(device)
    started = perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    _sync_if_cuda(device)
    ended = perf_counter()

    latency = ended - started
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    total_tokens = int(output_ids.shape[-1])
    completion_tokens = total_tokens - prompt_tokens

    completion_tps = completion_tokens / latency if latency > 0 else 0.0
    total_tps = total_tokens / latency if latency > 0 else 0.0

    gpu_peak_mem_mb = None
    if device.type == "cuda":
        gpu_peak_mem_mb = torch.cuda.max_memory_allocated(device) / (
            1024 * 1024
        )

    return InferenceStats(
        prompt_index=prompt_index,
        run_index=run_index,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_per_s=completion_tps,
        total_tokens_per_s=total_tps,
        gpu_peak_mem_mb=gpu_peak_mem_mb,
    )


def _summarize(rows: Sequence[InferenceStats]) -> dict[str, float]:
    return {
        "latency_s_avg": mean(r.latency_s for r in rows),
        "prompt_tokens_avg": mean(r.prompt_tokens for r in rows),
        "completion_tokens_avg": mean(r.completion_tokens for r in rows),
        "total_tokens_avg": mean(r.total_tokens for r in rows),
        "completion_tps_avg": mean(r.completion_tokens_per_s for r in rows),
        "total_tps_avg": mean(r.total_tokens_per_s for r in rows),
    }


def run_benchmark(
    *,
    message_sets: Sequence[Sequence[ChatMessage]],
    warmup_runs: int = 1,
    measured_runs: int = 3,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> list[InferenceStats]:
    all_rows: list[InferenceStats] = []

    for prompt_index, messages in enumerate(message_sets, start=1):
        for warmup_index in range(warmup_runs):
            _benchmark_once(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                prompt_index=prompt_index,
                run_index=-(warmup_index + 1),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        prompt_rows: list[InferenceStats] = []
        for run_index in range(1, measured_runs + 1):
            row = _benchmark_once(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                prompt_index=prompt_index,
                run_index=run_index,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            prompt_rows.append(row)
            all_rows.append(row)

            gpu = (
                f", gpu_peak_mem_mb={row.gpu_peak_mem_mb:.1f}"
                if row.gpu_peak_mem_mb is not None
                else ""
            )
            print(
                f"prompt={prompt_index} run={run_index} "
                f"latency_s={row.latency_s:.3f} "
                f"prompt_tokens={row.prompt_tokens} "
                f"completion_tokens={row.completion_tokens} "
                f"completion_tps={row.completion_tokens_per_s:.2f}{gpu}"
            )

    return all_rows


message_sets: list[list[ChatMessage]] = [
    [
        {
            "role": "user",
            "content": "Write a short explanation of gradient descent in simple terms.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Create a 5-step plan to debug a Python memory leak.",
        }
    ],
    [
        {
            "role": "user",
            "content": "Summarize why quantization helps model inference speed.",
        }
    ],
]

all_rows = run_benchmark(message_sets=message_sets)

for prompt_index, rows_iter in groupby(
    all_rows, key=lambda row: row.prompt_index
):
    prompt_summary = _summarize(list(rows_iter))
    print(f"summary prompt={prompt_index}: {prompt_summary}")

overall = _summarize(all_rows)
print(f"summary overall: {overall}")
