import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# =======================================================
# Utility
# =======================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_json(root_dir, name):
    results = []
    for root, dirs, files in os.walk(root_dir):
        if name in files:
            results.append(os.path.join(root, name))
    return results

def detect_method(path):
    if path.startswith("end_results"):
        return "earlystop"
    if path.startswith("recall_results"):
        return "normal"
    return "unknown"

def parse_dynamic_path(path):
    """
    自动解析路径:
      <root>/raw_results/Offset_results/<dataset>/<params>/<query_cnt>/<repeat>/HNSWIO_xxx.json
    """
    parts = path.split("/")

    # 找到文件名索引
    fname = os.path.basename(path)
    idx = parts.index(fname)

    # -----------------------
    # 自动识别 dataset / params / query_cnt / repeat
    # -----------------------
    dataset = None
    params = None
    query_cnt = None
    repeat = None

    # 我们倒着扫描文件路径
    # [ ... dataset / params / qcnt / repeat / file ]
    # 规则：
    #   repeat:    纯数字
    #   query_cnt: 纯数字
    #   params:    形如 "1_300_256"（必须三个段）
    #   dataset:   其他非数字字符串
    #
    p_repeat        = parts[idx - 1]
    p_query_cnt     = parts[idx - 2]
    p_params        = parts[idx - 3]
    p_dataset       = parts[idx - 4]

    # parse repeat
    if p_repeat.isdigit():
        repeat = int(p_repeat)
    else:
        raise ValueError(f"Cannot parse repeat in path: {path}")

    # parse query_cnt
    if p_query_cnt.isdigit():
        query_cnt = int(p_query_cnt)
    else:
        raise ValueError(f"Cannot parse query_cnt in path: {path}")

    # parse params block: must have 3 parts and all numeric
    param_parts = p_params.split("_")
    if len(param_parts) == 3 and all(x.isdigit() for x in param_parts):
        threads, ef, iodepth = map(int, param_parts)
    else:
        raise ValueError(f"Invalid param block '{p_params}' in path: {path}")

    # dataset 就是上一层目录，不需要限制（支持 deep, glove1M, fashion-mnist 等）
    dataset = p_dataset

    return dataset, threads, ef, iodepth, query_cnt, repeat


# =======================================================
# Collect all results
# =======================================================

def collect_all():
    rows = []
    roots = ["end_results", "recall_results"]

    for root in roots:
        # recall
        for f in find_json(root, "HNSWIO_Recall.json"):
            js = load_json(f)

            method = detect_method(f)
            dataset, th, ef, dpth, qcnt, rpt = parse_dynamic_path(f)

            rows.append({
                "method": method,
                "dataset": dataset,
                "threads": th,
                "ef": ef,
                "iodepth": dpth,
                "param_label": f"{method}(ef={ef},depth={dpth},thr={th})",
                "recall": js["avgcost"]["avg_recall"],
                "iter": None,
                "dist": None
            })

        # iter-dist
        for f in find_json(root, "HNSWIO_IterDistCount.json"):
            js = load_json(f)

            method = detect_method(f)
            dataset, th, ef, dpth, qcnt, rpt = parse_dynamic_path(f)

            rows.append({
                "method": method,
                "dataset": dataset,
                "threads": th,
                "ef": ef,
                "iodepth": dpth,
                "param_label": f"{method}(ef={ef},depth={dpth},thr={th})",
                "recall": None,
                "iter": js["avgcost"]["avg_iter_count"],
                "dist": js["avgcost"]["avg_dist_count"]
            })

    df = pd.DataFrame(rows)

    # merge recall + dist + iter
    df = df.groupby(
        ["dataset", "method", "ef", "iodepth", "threads", "param_label"],
        as_index=False
    ).agg({
        "recall": "max",
        "iter": "max",
        "dist": "max"
    })

    return df


# =======================================================
# Plot (3 subplots, param_label included)
# =======================================================
def draw_subplots(df, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    pdf = PdfPages(os.path.join(outdir, "method_dataset_subplots_fullparam.pdf"))

    df = df.sort_values(["dataset", "method", "ef"])
    datasets = df["dataset"].unique()
    x = range(len(datasets))

    fig, axes = plt.subplots(3, 1, figsize=(15, 16), sharex=True)

    labels = df["param_label"].unique()

    # --------------------------
    # (1) RECALL subplot
    # --------------------------
    ax = axes[0]
    for label in labels:
        sub = df[df["param_label"] == label]
        recalls = []

        for ds in datasets:
            if ds in sub["dataset"].values:
                v = sub[sub["dataset"] == ds]["recall"].values[0]
                recalls.append(v)
            else:
                recalls.append(None)

        # 全 None 的不画
        if all(v is None for v in recalls):
            continue

        # 过滤 None
        xx = [xi for xi, v in zip(x, recalls) if v is not None]
        yy = [v  for v  in recalls if v is not None]

        ax.plot(xx, yy, marker='o', linewidth=2, label=label)

    ax.set_title("Recall vs Dataset")
    ax.set_ylabel("Recall")
    ax.grid(True)
    ax.legend()


    # --------------------------
    # (2) ITER COUNT subplot
    # --------------------------
    ax = axes[1]
    width = 0.8 / len(labels)

    for i, label in enumerate(labels):
        sub = df[df["param_label"] == label]
        iters = []

        for ds in datasets:
            if ds in sub["dataset"].values:
                v = sub[sub["dataset"] == ds]["iter"].values[0]
                iters.append(v)
            else:
                iters.append(None)

        if all(v is None for v in iters):
            continue

        xx = []
        yy = []
        for xi, v in zip(x, iters):
            if v is not None:
                xx.append(xi - 0.4 + width*i)
                yy.append(v)

        ax.bar(xx, yy, width=width, label=label, alpha=0.7)

    ax.set_title("Iter Count vs Dataset")
    ax.set_ylabel("Iter Count")
    ax.grid(True)
    ax.legend()


    # --------------------------
    # (3) DIST COUNT subplot
    # --------------------------
    ax = axes[2]

    for label in labels:
        sub = df[df["param_label"] == label]
        dists = []

        for ds in datasets:
            if ds in sub["dataset"].values:
                v = sub[sub["dataset"] == ds]["dist"].values[0]
                dists.append(v)
            else:
                dists.append(None)

        if all(v is None for v in dists):
            continue

        xx = [xi for xi, v in zip(x, dists) if v is not None]
        yy = [v  for v  in dists if v is not None]

        ax.plot(xx, yy, marker='s', linestyle='--', linewidth=2, label=label)

    ax.set_title("Dist Count vs Dataset")
    ax.set_ylabel("Dist Count")
    ax.grid(True)
    ax.legend()


    # --------------------------
    # FINAL LAYOUT
    # --------------------------
    plt.xticks(list(x), datasets, rotation=45)
    plt.tight_layout()

    png_path = os.path.join(outdir, "method_dataset_subplots_fullparam.png")
    plt.savefig(png_path)
    pdf.savefig()
    plt.close()
    pdf.close()

    return png_path


# =======================================================
# Markdown
# =======================================================

def write_md(df, img_path, out="summary.md"):
    with open(out, "w") as f:
        f.write("# Early-Stop vs Normal (Full Parameter Comparison)\n\n")

        f.write("## Table\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Figures (3 Subplots with Different Parameters)\n")
        f.write(f"![comparison]({img_path})\n\n")


# =======================================================
# Main
# =======================================================

if __name__ == "__main__":
    print("Collecting results...")
    df = collect_all()

    print("Drawing subplots with param labels...")
    img = draw_subplots(df)

    print("Writing summary.md ...")
    write_md(df, img)

    print("Done!")
