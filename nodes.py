import json
from pathlib import Path

import numpy as np

from .optimizers import (
    create_optimizer,
    optimizer_from_state,
    METHOD_NAMES,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PACKAGE_ROOT / "logs"

def _sanitize_iter_name(name):
    name = str(name or "").strip()
    if not name:
        return "iter1"
    return name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _list_to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def _parse_lora_stack_entry(entry, bounds):
    if entry is None:
        return None
    name = ""
    weight = 0.0
    clip = 0.0
    if isinstance(entry, dict):
        name = str(entry.get("name") or entry.get("lora_name") or "").strip()
        weight = _list_to_float(entry.get("weight") or entry.get("strength") or 0.0)
        clip = _list_to_float(entry.get("clip") or entry.get("clipStrength") or weight)
    elif isinstance(entry, (list, tuple)):
        if not entry:
            return None
        name = str(entry[0]).strip()
        weight = _list_to_float(entry[1] if len(entry) > 1 else 0.0)
        clip = _list_to_float(entry[2] if len(entry) > 2 else weight)
    elif isinstance(entry, str):
        name = entry.strip()
        weight = 0.0
        clip = 0.0
    if not name:
        return None
    return {
        "name": name,
        "weight": weight,
        "clip": clip,
        "bounds": bounds,
    }


def _collect_lora_entries(lora_stack_neg1_1, lora_stack_0_1, lora_stack_0_2):
    entries = []
    for item in (lora_stack_neg1_1 or []):
        parsed = _parse_lora_stack_entry(item, [-1.0, 1.0])
        if parsed is not None:
            entries.append(parsed)
    for item in (lora_stack_0_1 or []):
        parsed = _parse_lora_stack_entry(item, [0.0, 1.0])
        if parsed is not None:
            entries.append(parsed)
    for item in (lora_stack_0_2 or []):
        parsed = _parse_lora_stack_entry(item, [0.0, 2.0])
        if parsed is not None:
            entries.append(parsed)
    return entries


def _parse_ranking_string(ranking, lam=None):
    if ranking is None:
        return []
    if isinstance(ranking, str):
        raw = ranking.strip()
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            ranks = [int(p) for p in parts]
        except ValueError:
            raise ValueError("Ranking format invalid: enter comma-separated integers like '3,1,2'.")
    elif isinstance(ranking, (list, tuple)):
        ranks = [int(item) for item in ranking if item is not None]
    else:
        raise ValueError("Ranking must be a comma-separated string or a list of integers.")

    if lam is None:
        lam = len(ranks)
    if len(ranks) != lam:
        raise ValueError(f"Ranking length must equal batch_size ({lam}).")
    if set(ranks) != set(range(1, lam + 1)):
        raise ValueError(f"Ranking must be a permutation of 1..{lam}.")
    return ranks


def _find_log(log_name):
    """按 log_name 精确查找，返回 (data, path) 或 (None, None)"""
    log_file = LOGS_DIR / log_name / "optimization_log.json"
    if not log_file.exists():
        return None, None
    try:
        with log_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None
    return data, log_file


class LoraWeightAutoOptimizer:
    NAME = "LoraOptimizer"
    CATEGORY = "Lora Weight Auto Optimizer"
    FUNCTION = "optimize"
    RETURN_TYPES = ("LORA_STACK", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_stack", "lora_stack_text", "status", "filename_prefix")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {"default": 0, "forceInput": True}),
                "batch_size": ("INT", {"default": 0, "forceInput": True}),
                "method": (METHOD_NAMES, {"default": "CMA-ES"}),
            },
            "optional": {
                "lora_stack_0_1": ("LORA_STACK",),
                "lora_stack_neg1_1": ("LORA_STACK",),
                "lora_stack_0_2": ("LORA_STACK",),
                "log_name": ("STRING", {"default": "iter1"}),
                "ranking": ("STRING", {"default": ""}),
            },
        }

    def optimize(
        self,
        index=0,
        batch_size=4,
        method="CMA-ES",
        lora_stack_neg1_1=None,
        lora_stack_0_1=None,
        lora_stack_0_2=None,
        ranking=None,
        log_name="iter1",
    ):
        index = max(0, int(index))
        batch_size = max(2, int(batch_size))
        method = str(method).strip()
        log_name = _sanitize_iter_name(log_name)

        # ── 1. 排名合法性校验（最先执行，不合法直接报错） ──
        normalized_ranking = []
        ranking_provided = isinstance(ranking, str) and ranking.strip()
        if ranking_provided:
            normalized_ranking = _parse_ranking_string(ranking, batch_size)

        mean_change = 0.0  # 仅 update 后有意义

        # ── 2. 收集 LoRA 条目 ──
        entries = _collect_lora_entries(
            lora_stack_neg1_1,
            lora_stack_0_1,
            lora_stack_0_2,
        )
        if not entries:
            return ([], "", "no_loras", "")

        lora_names = [entry["name"] for entry in entries]
        bounds = [entry["bounds"] for entry in entries]
        input_entries = [
            {
                "name": entry["name"],
                "weight": float(entry["weight"]),
                "clip": float(entry["clip"]),
                "bounds": entry["bounds"],
            }
            for entry in entries
        ]

        # ── 3. 按 log_name 查找已有 log ──
        log_state, resolved_log = _find_log(log_name)

        # ═══════════════════════════════════════════════════════
        #  分支 A：没有找到同名 log → 新建，无视 ranking
        # ═══════════════════════════════════════════════════════
        if log_state is None:
            optimizer = create_optimizer(method, len(bounds), bounds, batch_size=batch_size)
            optimizer.reset()
            candidates = optimizer.sample_batch()
            gen_display = 1
            status_str = f"initialized:{log_name}_1-{index + 1} σ=0.2000"

        else:
            # ═══════════════════════════════════════════════════
            #  分支 B：找到同名 log → 校验一致性
            # ═══════════════════════════════════════════════════
            log_lora_names = log_state.get("lora_names", [])
            log_batch_size = log_state.get("batch_size")
            log_method = log_state.get("method", "")

            if log_method and log_method != method:
                raise ValueError(
                    f"Method mismatch with existing log '{log_name}'. "
                    f"Log has: '{log_method}', current: '{method}'"
                )
            if log_lora_names != lora_names:
                raise ValueError(
                    f"LoRA list mismatch with existing log '{log_name}'. "
                    f"Log has: {log_lora_names}, current: {lora_names}"
                )
            if log_batch_size is not None and int(log_batch_size) != batch_size:
                raise ValueError(
                    f"batch_size ({batch_size}) does not match existing log "
                    f"'{log_name}' ({log_batch_size})."
                )
            
            # 恢复 optimizer + current_z / current_y
            opt_state = log_state.get("optimizer_state")
            if not opt_state:
                raise ValueError(f"Log '{log_name}' is missing optimizer_state.")
            optimizer = optimizer_from_state(opt_state)
            saved_z = log_state.get("current_z", [])
            saved_y = log_state.get("current_y", [])
            if saved_z:
                optimizer.current_z = saved_z
            if saved_y:
                optimizer.current_y = saved_y
            gen_display = optimizer.gen + 1  # gen=0 → 第1轮

            if normalized_ranking and index == 0:
                # ── B1：排名合法且 index=0 → 结算当前代，采样下一代 ──
                # 只有 index=0 执行 update，防止同一 queue 中多节点重复更新
                mean_before = optimizer.mean.copy()
                optimizer.update(normalized_ranking)
                candidates = optimizer.sample_batch()
                gen_display = optimizer.gen + 1  # update 后 gen 已递增
                mean_change = float(np.linalg.norm(optimizer.mean - mean_before))
                sigma = float(optimizer.sigma)
                # 收敛判定
                if sigma < 0.005:
                    conv_hint = " (converged)"
                elif sigma < 0.02:
                    conv_hint = " (near)"
                else:
                    conv_hint = ""
                # 覆盖写入同一 iter 目录
                resolved_log = None
                status_str = f"index={index} ranking={normalized_ranking} \n updated:{log_name}_{gen_display}-{index + 1} σ={sigma:.4f} Δm={mean_change:.4f}{conv_hint}"
            else:
                # ── B2：排名为空 → 从 log 的 all_candidates 重建当前代候选 ──
                saved_candidates = log_state.get("all_candidates")
                if saved_candidates and len(saved_candidates) == optimizer.lam:
                    candidates = [np.array(c, dtype=float) for c in saved_candidates]
                else:
                    # 降级：重新采样
                    candidates = optimizer.sample_batch()
                saved_mean_change = log_state.get("mean_change", 0.0)
                status_str = f"index={index} ranking={normalized_ranking} \n resumed:{log_name}_{gen_display}-{index + 1} σ={optimizer.sigma:.4f} Δm={saved_mean_change:.4f}"

        # ── 4. 选出当前 index 对应的候选 ──
        candidate_index = max(0, min(index, len(candidates) - 1))
        selected = candidates[candidate_index]

        # ── 5. 构建 lora_stack 输出 ──
        lora_stack = [
            (
                lora_names[i] if i < len(lora_names) else f"lora_{i}",
                float(selected[i]),
                float(selected[i]),
            )
            for i in range(len(selected))
        ]

        # ── 6. 保存 log ──
        estimated_weights = optimizer.mean.tolist()
        sigma = float(optimizer.sigma)
        mean_change = mean_change if normalized_ranking and index == 0 else (
            log_state.get("mean_change", 0.0) if log_state else 0.0
        )
        log_data = {
            "iter_name": log_name,
            "log_name": log_name,
            "method": method,
            "batch_size": int(optimizer.lam),
            "last_index": index,
            "candidate_index": int(candidate_index),
            "generation": int(optimizer.gen),
            "sigma": sigma,
            "mean_change": mean_change,
            "lora_names": lora_names,
            "bounds": bounds,
            "input_lora_stack": input_entries,
            "estimated_weights": [float(x) for x in estimated_weights],
            "current_batch_weights": [float(x) for x in selected],
            "current_batch": [
                {"name": lora_names[i] if i < len(lora_names) else f"lora_{i}", "weight": float(selected[i])}
                for i in range(len(selected))
            ],
            "all_candidates": [[float(x) for x in c] for c in candidates],
            "optimizer_state": optimizer.get_state(),
            "current_z": optimizer.current_z,
            "current_y": optimizer.current_y,
        }

        save_dir = resolved_log.parent if resolved_log is not None else LOGS_DIR / log_name
        save_dir.mkdir(parents=True, exist_ok=True)
        log_file = save_dir / "optimization_log.json"
        try:
            with log_file.open("w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # ── 7. 格式化输出文本 ──
        try:
            candidate_lines = "\n".join([
                f"{lora_names[i] if i < len(lora_names) else f'lora_{i}'},estimated:{float(estimated_weights[i]):.2f},current:{float(w):.2f}"
                for i, (_, w, _) in enumerate(lora_stack)
            ])
            weight_str = f"candidate:\n{candidate_lines}"
        except Exception:
            weight_str = ""

        weights_str = "_".join(f"{float(w):.2f}" for w in selected)
        filename_prefix = f"{log_name}/{log_name}_{gen_display}-{index + 1}_{weights_str}"

        return (lora_stack, weight_str, status_str, filename_prefix)


NODE_CLASS_MAPPINGS = {
    LoraWeightAutoOptimizer.NAME: LoraWeightAutoOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    LoraWeightAutoOptimizer.NAME: "Lora Weight Auto Optimizer",
}
