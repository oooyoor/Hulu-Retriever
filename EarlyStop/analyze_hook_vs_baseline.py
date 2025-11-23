import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def analyze_one_setting(
    dataset: str,
    num_threads: int,
    ef_search: int,
    io_depth: int,
    query_cnt: int,
    repeat_id: int,
) -> Dict[str, Any]:
    base_dir = "/home/zqf/Hulu-Retriever/SearchDifficultyResults"

    base_out_dir = os.path.join(
        base_dir,
        "BaselineResults",
        dataset,
        f"{num_threads}_{ef_search}_{io_depth}",
        str(query_cnt),
        str(repeat_id),
    )
    hook_out_dir = os.path.join(
        base_dir,
        "Hook_results",
        dataset,
        f"{num_threads}_{ef_search}_{io_depth}",
        str(query_cnt),
        str(repeat_id),
    )

    base_iter_path = os.path.join(base_out_dir, "HNSWIO_IterDistCount.json")
    base_recall_path = os.path.join(base_out_dir, "HNSWIO_Recall.json")
    hook_recall_path = os.path.join(hook_out_dir, "HNSWIO_Recall.json")
    hook_iter_path = os.path.join(hook_out_dir, "HNSWIO_IterDistCount.json")

    print(f"[INFO] Baseline iter/dist: {base_iter_path}")
    print(f"[INFO] Baseline recall   : {base_recall_path}")
    print(f"[INFO] Hook iter/dist    : {hook_iter_path}")
    print(f"[INFO] Hook recall       : {hook_recall_path}")

    base_iter_json = load_json(base_iter_path)
    base_recall_json = load_json(base_recall_path)
    hook_iter_json = load_json(hook_iter_path)
    hook_recall_json = load_json(hook_recall_path)

    base_iter_entries = base_iter_json["entries"]
    base_recall_entries = base_recall_json["entries"]
    hook_iter_entries = hook_iter_json["entries"]
    hook_recall_entries = hook_recall_json["entries"]

    n = min(
        len(base_iter_entries),
        len(base_recall_entries),
        len(hook_iter_entries),
        len(hook_recall_entries),
        query_cnt,
    )

    print(f"[INFO] Using first {n} queries for analysis")

    improved_indices: List[int] = []
    degraded_indices: List[int] = []
    unchanged_indices: List[int] = []

    stats = {
        "delta_recall": [],
        "speedup_iter": [],
        "speedup_dist": [],
    }

    for i in range(n):
        recall_base = float(base_recall_entries[i].get("recall", 0.0))
        recall_hook = float(hook_recall_entries[i].get("recall", 0.0))

        iter_base = float(base_iter_entries[i].get("iter_count", 0.0))
        dist_base = float(base_iter_entries[i].get("dist_count", 0.0))

        iter_hook = float(hook_iter_entries[i].get("iter_count", 0.0))
        dist_hook = float(hook_iter_entries[i].get("dist_count", 0.0))

        delta_recall = recall_hook - recall_base
        speedup_iter = iter_base / max(1.0, iter_hook)
        speedup_dist = dist_base / max(1.0, dist_hook)

        stats["delta_recall"].append(delta_recall)
        stats["speedup_iter"].append(speedup_iter)
        stats["speedup_dist"].append(speedup_dist)

        # 新的分类标准（优先基于 recall，然后看 iter_count）:
        # - Good (Improved): recall 没有下降 (delta_recall >= 0)
        # - Mid1 (Mixed): recall 下降但 iter_count 更小 (delta_recall < 0 且 speedup_iter > 1.0)
        # - Bad (Degraded): recall 下降且 iter_count 增加或不变 (delta_recall < 0 且 speedup_iter <= 1.0)
        if delta_recall >= 0:
            improved_indices.append(i)
        elif delta_recall < 0 and speedup_iter > 1.0:
            unchanged_indices.append(i)  # Mid1: recall 下降但有加速
        else:
            degraded_indices.append(i)  # Bad: recall 下降且没有加速

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    avg_delta = avg(stats["delta_recall"])
    avg_sp_iter = avg(stats["speedup_iter"])
    avg_sp_dist = avg(stats["speedup_dist"])

    print("\n====== Overall Stats ======")
    print(f"Total queries       : {n}")
    print(f"Good (Improved)     : {len(improved_indices)}  (recall 没有下降)")
    print(f"Mid1 (Mixed)        : {len(unchanged_indices)}  (recall 下降但 iter_count 更小)")
    print(f"Bad (Degraded)      : {len(degraded_indices)}  (recall 下降且 iter_count 增加或不变)")
    print("")
    print(f"Avg Δrecall         : {avg_delta:.6f}")
    print(f"Avg speedup_iter    : {avg_sp_iter:.4f}")
    print(f"Avg speedup_dist    : {avg_sp_dist:.4f}")

    # 根据新的分类标准判断：
    # GOOD: recall 没有下降（avg_delta >= 0）
    # BAD: recall 下降且没有加速（avg_delta < 0 且 speedup <= 1）
    # NEUTRAL: recall 下降但有加速（avg_delta < 0 但 speedup > 1）
    verdict_short = "MID1"
    verdict_msg = "MID1: recall 下降但有加速，存在 trade-off。"
    
    if avg_delta >= 0:
        verdict_short = "GOOD"
        verdict_msg = "GOOD: recall 没有下降（精度保持或提升）。"
    elif avg_delta < 0 and avg_sp_iter <= 1.0 and avg_sp_dist <= 1.0:
        verdict_short = "BAD"
        verdict_msg = "BAD: recall 下降且没有加速（既掉精度又没提速）。"
    else:
        # avg_delta < 0 但至少有一个 speedup > 1.0
        verdict_short = "MID1"
        verdict_msg = "MID1: recall 下降但有加速，存在 trade-off。"
    
    verdict = verdict_msg

    print(f"\n[VERDICT] {verdict}")

    print("\n====== Example Good Queries (up to 20) ======")
    for idx in improved_indices[:20]:
        print(
            f"q={idx:6d}  Δrecall={stats['delta_recall'][idx]:+.6f}  "
            f"speedup_iter={stats['speedup_iter'][idx]:.3f}  "
            f"speedup_dist={stats['speedup_dist'][idx]:.3f}"
        )

    print("\n====== Example Mid1 Queries (up to 20) ======")
    for idx in unchanged_indices[:20]:
        print(
            f"q={idx:6d}  Δrecall={stats['delta_recall'][idx]:+.6f}  "
            f"speedup_iter={stats['speedup_iter'][idx]:.3f}  "
            f"speedup_dist={stats['speedup_dist'][idx]:.3f}"
        )

    print("\n====== Example Bad Queries (up to 20) ======")
    for idx in degraded_indices[:20]:
        print(
            f"q={idx:6d}  Δrecall={stats['delta_recall'][idx]:+.6f}  "
            f"speedup_iter={stats['speedup_iter'][idx]:.3f}  "
            f"speedup_dist={stats['speedup_dist'][idx]:.3f}"
        )

    # 保存详细结果到一个 JSON 文件
    out_detail_path = os.path.join(hook_out_dir, "HNSWIO_vsBaseline_python_analysis.json")
    detail = {
        "meta": {
            "dataset": dataset,
            "num_threads": num_threads,
            "search_ef": ef_search,
            "io_depth": io_depth,
            "query_cnt": n,
            "repeat_id": repeat_id,
        },
        "per_query": [
            {
                "query_id": i,
                "delta_recall": stats["delta_recall"][i],
                "speedup_iter": stats["speedup_iter"][i],
                "speedup_dist": stats["speedup_dist"][i],
                "label": (
                    "good"
                    if i in improved_indices
                    else "bad"
                    if i in degraded_indices
                    else "mid1"
                ),
            }
            for i in range(n)
        ],
    }
    with open(out_detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"\n[INFO] Detailed per-query analysis saved to: {out_detail_path}")

    # 如果设置了环境变量 HOOK_MD_REPORT，则把关键信息追加到该 markdown 文件中，方便总览
    md_path: Optional[str] = os.environ.get("HOOK_MD_REPORT")
    if md_path:
        os.makedirs(os.path.dirname(md_path), exist_ok=True)
        with open(md_path, "a") as mf:
            mf.write(f"### {dataset} | threads={num_threads}, ef={ef_search}, "
                     f"iodepth={io_depth}, query={n}, repeat={repeat_id}\n\n")
            mf.write(f"- **Avg Δrecall**: {avg_delta:.6f}\n")
            mf.write(f"- **Avg speedup_iter**: {avg_sp_iter:.4f}\n")
            mf.write(f"- **Avg speedup_dist**: {avg_sp_dist:.4f}\n")
            mf.write(f"- **Good / Mid1 / Bad**: "
                     f"{len(improved_indices)} / {len(unchanged_indices)} / {len(degraded_indices)}\n")
            mf.write(f"- **Verdict**: {verdict}\n\n")

    return {
        "dataset": dataset,
        "threads": num_threads,
        "ef": ef_search,
        "iodepth": io_depth,
        "query_cnt": n,
        "repeat_id": repeat_id,
        "avg_delta_recall": avg_delta,
        "avg_speedup_iter": avg_sp_iter,
        "avg_speedup_dist": avg_sp_dist,
        "good": len(improved_indices),
        "mid1": len(unchanged_indices),
        "bad": len(degraded_indices),
        "verdict": verdict,
        "verdict_short": verdict_short,  # 用于分类
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Hook (early-stop) vs Baseline results per-query."
    )
    parser.add_argument("--dataset", type=str, required=True, help="e.g., sift")
    parser.add_argument("--threads", type=int, required=True, help="num_threads")
    parser.add_argument("--ef", type=int, required=True, help="search_ef")
    parser.add_argument("--iodepth", type=int, required=True, help="io_depth")
    parser.add_argument("--query", type=int, required=True, help="query_cnt")
    parser.add_argument("--repeat", type=int, default=1, help="repeat_id")
    parser.add_argument("--output-json", type=str, help="Optional: append result to this JSON array file")

    args = parser.parse_args()

    result = analyze_one_setting(
        dataset=args.dataset,
        num_threads=args.threads,
        ef_search=args.ef,
        io_depth=args.iodepth,
        query_cnt=args.query,
        repeat_id=args.repeat,
    )

    # 如果指定了输出 JSON 文件，追加结果
    if args.output_json:
        try:
            if os.path.exists(args.output_json):
                with open(args.output_json, "r") as f:
                    all_results = json.load(f)
            else:
                all_results = []
            all_results.append(result)
            with open(args.output_json, "w") as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write to {args.output_json}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()


