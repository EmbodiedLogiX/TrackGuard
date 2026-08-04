from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq

OPTION_LETTERS = list("ABCDEFGHIJ")


def coerce_answer(text: str, valid: set[str]) -> str | None:
    if not text:
        return None
    for m in re.finditer(r"[A-J]", text.strip().upper()):
        ch = m.group(0)
        if ch in valid:
            return ch
    return None


def load_model(base_model: str, lora_path: str | None):
    processor = AutoProcessor.from_pretrained(
        base_model, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForVision2Seq.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return processor, model


def build_messages(image_paths: list[str], question_text: str) -> list[dict]:
    content = [{"type": "image", "image": Image.open(p).convert("RGB")}
               for p in image_paths]
    content.append({"type": "text", "text": question_text})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def predict_one(processor, model, image_paths: list[str], question_text: str,
                max_new_tokens: int) -> tuple[str, dict]:
    cuda = torch.cuda.is_available()

    t_pre0 = time.perf_counter()
    messages = build_messages(image_paths, question_text)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    if cuda:
        torch.cuda.synchronize()
    t_gen0 = time.perf_counter()

    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if cuda:
        torch.cuda.synchronize()
    t_gen1 = time.perf_counter()

    n_prompt_tokens = int(inputs.input_ids.shape[1])
    n_new_tokens = int(out_ids.shape[1] - n_prompt_tokens)
    raw = processor.batch_decode(
        out_ids[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0].strip()

    timing = {
        "preprocess_s": t_gen0 - t_pre0,
        "generate_s": t_gen1 - t_gen0,
        "total_s": t_gen1 - t_pre0,
        "prompt_tokens": n_prompt_tokens,
        "new_tokens": n_new_tokens,
    }
    return raw, timing


def _mb(nbytes: int) -> float:
    return nbytes / (1024 ** 2)


def _sum_allocated_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return sum(_mb(torch.cuda.memory_allocated(i)) for i in range(torch.cuda.device_count()))


def _reset_peak_mem() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def _peak_mem_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = sum(_mb(torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count()))
    reserved = sum(_mb(torch.cuda.max_memory_reserved(i)) for i in range(torch.cuda.device_count()))
    return alloc, reserved


def _device_total_used_mb() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    used = total = 0.0
    for i in range(torch.cuda.device_count()):
        free_b, total_b = torch.cuda.mem_get_info(i)
        used += _mb(total_b - free_b)
        total += _mb(total_b)
    return used, total


def print_metrics(rows: list[dict], mem: dict | None = None) -> None:
    total = len(rows)
    n_parse_fail = sum(1 for r in rows if r["pred"] is None)
    correct = sum(1 for r in rows if r["pred"] is not None and r["pred"] == r["gt"])
    acc = correct / total if total else 0.0

    print("\n" + "=" * 60)
    print("recovery MCQ test-set evaluation")
    print("=" * 60)
    print(f"  samples:            {total}")
    print(f"  unparsed (invalid): {n_parse_fail}")
    print(f"  overall accuracy:   {acc:.4f} ({correct}/{total})")

    if total:
        rand_exp = sum(1.0 / max(1, r["n_options"]) for r in rows) / total
        print(f"  random-guess acc:   {rand_exp:.4f}")

    by_gt: dict[str, list] = {}
    for r in rows:
        by_gt.setdefault(r["gt"], []).append(r)
    print(f"\n  accuracy by ground-truth answer (gt):")
    for k in sorted(by_gt.keys()):
        sub = by_gt[k]
        c = sum(1 for r in sub if r["pred"] == r["gt"])
        print(f"    gt={k:>3}: {c}/{len(sub)} = {c/len(sub):.4f}")

    by_nopt: dict[int, list] = {}
    for r in rows:
        by_nopt.setdefault(r["n_options"], []).append(r)
    print(f"\n  accuracy by number of options (n_options):")
    for k in sorted(by_nopt.keys()):
        sub = by_nopt[k]
        c = sum(1 for r in sub if r["pred"] == r["gt"])
        print(f"    n_options={k}: {c}/{len(sub)} = {c/len(sub):.4f}")

    lat = sorted(r["latency_s"] for r in rows if r.get("latency_s") is not None)
    gen = [r["generate_s"] for r in rows if r.get("generate_s") is not None]
    new_tok = [r["new_tokens"] for r in rows if r.get("new_tokens") is not None]

    def _pct(vals: list[float], q: float) -> float:
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return vals[idx]

    n_timed = len(lat)
    mean_lat = sum(lat) / n_timed if n_timed else 0.0
    mean_gen = sum(gen) / len(gen) if gen else 0.0
    total_new_tok = sum(new_tok) if new_tok else 0
    total_gen_time = sum(gen) if gen else 0.0
    tok_per_s = (total_new_tok / total_gen_time) if total_gen_time else 0.0
    if n_timed:
        print(f"\n  latency (per-sample inference, no model load, CUDA-synced; {n_timed} samples):")
        print(f"    mean latency:     {mean_lat*1000:8.1f} ms  "
              f"(throughput {1.0/mean_lat if mean_lat else 0:.2f} samples/s)")
        print(f"    of which generate:{mean_gen*1000:8.1f} ms")
        print(f"    P50 / P90 / P99:  {_pct(lat,0.50)*1000:.1f} / "
              f"{_pct(lat,0.90)*1000:.1f} / {_pct(lat,0.99)*1000:.1f} ms")
        print(f"    min / max:        {lat[0]*1000:.1f} / {lat[-1]*1000:.1f} ms")
        print(f"    decode speed:     {tok_per_s:.1f} tokens/s  "
              f"(avg {total_new_tok/len(new_tok) if new_tok else 0:.1f} new tokens/sample)")

    if mem:
        if mem.get("gpu_available"):
            print(f"\n  GPU memory (summed over {mem.get('num_gpus', 0)} GPUs):")
            print(f"    model weights:    {mem['weights_mb']:8.1f} MB  ({mem['weights_mb']/1024:.2f} GB)")
            print(f"    peak allocated:   {mem['peak_alloc_mb']:8.1f} MB  ({mem['peak_alloc_mb']/1024:.2f} GB)")
            print(f"      infer overhead: {mem['infer_overhead_mb']:8.1f} MB")
            print(f"    peak reserved:    {mem['peak_reserved_mb']:8.1f} MB  ({mem['peak_reserved_mb']/1024:.2f} GB)")
            print(f"    device used/total:{mem['device_used_mb']:8.1f} / {mem['device_total_mb']:.1f} MB")
        else:
            print(f"\n  GPU memory: no GPU available (CPU inference, not measured)")
    print("=" * 60)


def load_samples(data_json: str) -> list[dict]:
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for e in data:
        images = e.get("images", [])
        user_msg = next((m for m in e["messages"] if m["role"] == "user"), None)
        asst_msg = next((m for m in e["messages"] if m["role"] == "assistant"), None)
        if user_msg is None or asst_msg is None or not images:
            continue
        question = user_msg["content"].replace("<image>", "").lstrip("\n")
        meta = e.get("_meta", {})
        gt = str(meta.get("answer", asst_msg["content"])).strip()
        samples.append({
            "images": images,
            "question": question,
            "gt": gt,
            "n_options": int(meta.get("n_options", 0)),
            "sample_id": meta.get("sample_id", ""),
            "sequence": meta.get("sequence", ""),
            "frame": meta.get("frame", ""),
            "history_frame": meta.get("history_frame", ""),
            "gt_id": meta.get("gt_id", ""),
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="checkpoints/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--lora_path", type=str, default="runs/recovery_vlm_lora")
    parser.add_argument("--data_json", type=str,
                        default="data/recovery_mcq_dataset/llama_factory/test.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--out_csv", type=str, default="runs/recovery_vlm_eval/predictions.csv")
    args = parser.parse_args()

    samples = load_samples(args.data_json)
    if args.limit and args.limit > 0:
        samples = samples[:args.limit]

    print(f"test set: {args.data_json}  ({len(samples)} samples)")
    print(f"   gt distribution: {Counter(s['gt'] for s in samples)}")
    lora = None if args.no_lora else args.lora_path
    print(f"   base_model: {args.base_model}")
    print(f"   lora_path:  {lora if lora else '(none; evaluating base model)'}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_before_mb = _sum_allocated_mb()

    processor, model = load_model(args.base_model, lora)

    weights_mb = max(0.0, _sum_allocated_mb() - mem_before_mb)
    _reset_peak_mem()

    rows: list[dict] = []
    t0 = time.time()
    for i, s in enumerate(samples):
        n_opts = s["n_options"] or len(OPTION_LETTERS)
        valid = set(OPTION_LETTERS[:n_opts])
        try:
            raw, timing = predict_one(
                processor, model, s["images"], s["question"], args.max_new_tokens,
            )
            pred = coerce_answer(raw, valid)
        except Exception as e:
            raw, pred = f"ERROR: {e}", None
            timing = {"preprocess_s": None, "generate_s": None,
                      "total_s": None, "prompt_tokens": None, "new_tokens": None}

        rows.append({
            "sample_id": s["sample_id"],
            "sequence": s["sequence"],
            "frame": s["frame"],
            "history_frame": s["history_frame"],
            "gt_id": s["gt_id"],
            "n_options": s["n_options"],
            "gt": s["gt"],
            "pred": pred,
            "correct": (pred is not None and pred == s["gt"]),
            "raw": raw,
            "preprocess_s": timing["preprocess_s"],
            "generate_s": timing["generate_s"],
            "latency_s": timing["total_s"],
            "prompt_tokens": timing["prompt_tokens"],
            "new_tokens": timing["new_tokens"],
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(samples):
            done = i + 1
            acc_so_far = sum(1 for r in rows if r["correct"]) / done
            rate = done / (time.time() - t0)
            print(f"  [{done}/{len(samples)}] running acc={acc_so_far:.4f}  ({rate:.2f} it/s)")

    peak_alloc_mb, peak_reserved_mb = _peak_mem_mb()
    dev_used_mb, dev_total_mb = _device_total_used_mb()
    mem = {
        "gpu_available": torch.cuda.is_available(),
        "num_gpus": (torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "weights_mb": weights_mb,
        "peak_alloc_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_reserved_mb,
        "infer_overhead_mb": max(0.0, peak_alloc_mb - weights_mb),
        "device_used_mb": dev_used_mb,
        "device_total_mb": dev_total_mb,
    }

    print_metrics(rows, mem)

    import os
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"\nper-sample predictions saved: {args.out_csv}")


if __name__ == "__main__":
    main()
