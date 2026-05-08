#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate NeuralConf trace JSONL files with an OpenAI-compatible endpoint.

Each output line is one sampled reasoning trace. The script stores the final
answer, the token-level confidence trajectory, and enough metadata to group
multiple traces belonging to the same question.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from openai import OpenAI

from neuralconf.evaluator import math_equal, strip_string


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def first_present(sample: Dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = sample.get(key)
        if value is not None and str(value).strip() != "":
            return str(value)
    return default


def format_choices(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        return "\n".join(f"{k}. {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        lines = []
        for i, item in enumerate(value):
            label = chr(ord("A") + i)
            lines.append(f"{label}. {item}")
        return "\n".join(lines)
    return str(value)


def build_prompt(sample: Dict[str, Any], args: argparse.Namespace) -> str:
    if args.prompt_field:
        prompt = sample.get(args.prompt_field)
        if prompt is None:
            raise KeyError(f"Prompt field '{args.prompt_field}' is missing in sample: {sample}")
        return str(prompt)

    context = first_present(sample, ["context", "passage", "article"])
    question = first_present(sample, ["question", "problem", "prompt"])
    choices = format_choices(sample.get("choices", sample.get("answers", "")))

    sections: List[str] = []
    if context:
        sections.append(str(context).strip())
    if question:
        sections.append(f"Question:\n{question.strip()}")
    if choices:
        sections.append(f"Choices:\n{choices.strip()}")

    sections.append(
        "Solve the problem step by step. Put the final answer inside \\boxed{}; "
        "the boxed content should contain only the final answer."
    )
    return "\n\n".join(sections).strip()


def last_boxed_only_string(text: str) -> Optional[str]:
    if text is None:
        return None
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
    if start < 0:
        return None

    left = text.find("{", start)
    if left < 0:
        return None

    depth = 0
    for idx in range(left, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start: idx + 1]
    return None


def remove_boxed(boxed: str) -> str:
    left = boxed.find("{")
    if left < 0 or not boxed.endswith("}"):
        return ""
    return boxed[left + 1: -1]


def extract_boxed_answer(text: str) -> str:
    boxed = last_boxed_only_string(text or "")
    if not boxed:
        return ""
    return strip_string(remove_boxed(boxed)).strip()


def compute_confidence(top_logprobs: Any) -> List[float]:
    """Convert top-logprob dictionaries into scalar token confidence values.

    For each generated token, this implementation stores the negative mean of
    the returned top-logprob values. This matches the confidence trajectory used
    by the accompanying NeuralConf training code.
    """
    confidences: List[float] = []
    if not top_logprobs:
        return confidences

    for token_dict in top_logprobs:
        if not token_dict:
            continue
        values = list(token_dict.values())
        if values:
            confidences.append(round(float(-np.mean(values)), 3))
    return confidences


def call_completion_api(client: OpenAI, prompt: str, args: argparse.Namespace, n: int):
    extra_body = {}
    if args.top_k_sampling > 0:
        extra_body["top_k"] = args.top_k_sampling

    kwargs = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "logprobs": args.logprobs,
        "n": n,
        "timeout": args.timeout,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    return client.completions.create(**kwargs)


def main(args: argparse.Namespace) -> None:
    if args.overwrite and os.path.exists(args.output_jsonl):
        os.remove(args.output_jsonl)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    dataset = load_jsonl(args.input_jsonl)

    for q_index, sample in enumerate(dataset, start=args.qid_start):
        qid = int(sample.get("qid", q_index))
        question = first_present(sample, ["question", "problem", "prompt"])
        prompt = build_prompt(sample, args)
        gold = first_present(sample, [args.gold_field, "gold_answer", "answer"], default="")
        gold = strip_string(gold).strip() if gold else ""

        valid_records: List[Dict[str, Any]] = []
        round_id = 0
        print("=" * 90)
        print(f"QID {qid}: collecting {args.total_budget} valid traces")
        print("=" * 90)

        while len(valid_records) < args.total_budget:
            round_id += 1
            if round_id > args.max_rounds_per_q:
                raise RuntimeError(
                    f"QID {qid}: exceeded --max-rounds-per-q={args.max_rounds_per_q}; "
                    f"collected {len(valid_records)}/{args.total_budget} valid traces."
                )

            remaining = args.total_budget - len(valid_records)
            request_n = remaining if args.request_batch_size <= 0 else min(remaining, args.request_batch_size)
            print(f"[QID {qid}] round {round_id}: requesting n={request_n}")

            start = time.time()
            response = call_completion_api(client, prompt, args, n=request_n)
            gen_time = time.time() - start

            kept = 0
            for choice in getattr(response, "choices", []):
                completion = getattr(choice, "text", "") or ""
                pred = extract_boxed_answer(completion)
                if not pred:
                    continue

                logprobs = getattr(choice, "logprobs", None)
                top_logprobs = getattr(logprobs, "top_logprobs", []) if logprobs else []
                confidences = compute_confidence(top_logprobs)

                record: Dict[str, Any] = {
                    "qid": qid,
                    "question": question,
                    "gold_answer": gold,
                    "pred_answer": pred,
                    "confidences": confidences,
                    "trace_id": len(valid_records),
                    "gen_time": round(gen_time, 2),
                }
                if gold:
                    record["is_correct"] = int(math_equal(str(pred), str(gold)))
                if args.save_prompt:
                    record["prompt"] = prompt
                if args.save_completion:
                    record["completion"] = completion

                valid_records.append(record)
                kept += 1
                if len(valid_records) >= args.total_budget:
                    break

            print(f"[QID {qid}] round {round_id}: kept {kept}, total {len(valid_records)}/{args.total_budget}")

        append_jsonl(args.output_jsonl, valid_records)

    print(f"Done. Results saved to {args.output_jsonl}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NeuralConf trace JSONL files with an OpenAI-compatible completions endpoint."
    )
    parser.add_argument("--input-jsonl", required=True, help="Input examples JSONL.")
    parser.add_argument("--output-jsonl", required=True, help="Output trace JSONL.")
    parser.add_argument("--model", required=True, help="Model name served by the endpoint.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--prompt-field", default="", help="Use this field directly as the prompt when provided.")
    parser.add_argument("--gold-field", default="answer", help="Gold-answer field in the input JSONL.")
    parser.add_argument("--qid-start", type=int, default=1)
    parser.add_argument("--total-budget", type=int, default=128, help="Number of valid traces to keep per question.")
    parser.add_argument("--request-batch-size", type=int, default=0, help="Completions requested per API call. 0 means all remaining traces.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k-sampling", type=int, default=40)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=20000)
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds.")
    parser.add_argument("--max-rounds-per-q", type=int, default=80)
    parser.add_argument("--save-prompt", action="store_true")
    parser.add_argument("--save-completion", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output file before writing.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
