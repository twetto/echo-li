"""Sweep affine Hessian-condition gates using paired affine KLT diagnostics."""

import argparse
import csv
from pathlib import Path

import numpy as np


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "seq": path.stem.replace("paired_affine_klt_diag_", ""),
                "err_translation": float(row["err_translation"]),
                "err_affine": float(row["err_affine"]),
                "err_delta": float(row["err_delta"]),
                "center_delta": float(row["center_delta"]),
                "quality_delta": float(row["quality_delta"]),
                "affine_h_cond": float(row["affine_h_cond"]),
                "affine_h_min": float(row["affine_h_min"]),
            })
    return rows


def summarize(rows, thresholds):
    finite = [
        r for r in rows
        if np.isfinite(r["err_translation"])
        and np.isfinite(r["err_affine"])
        and np.isfinite(r["affine_h_cond"])
    ]
    out = []
    for th in thresholds:
        sel = [r for r in finite if r["affine_h_cond"] <= th]
        if not sel:
            continue
        de = np.array([r["err_delta"] for r in sel], float)
        et = np.array([r["err_translation"] for r in sel], float)
        ea = np.array([r["err_affine"] for r in sel], float)
        center = np.array([r["center_delta"] for r in sel], float)
        out.append({
            "threshold": th,
            "accepted": len(sel),
            "accepted_frac": len(sel) / max(len(finite), 1),
            "median_delta": float(np.median(de)),
            "mean_delta": float(np.mean(de)),
            "p90_delta": float(np.percentile(de, 90)),
            "affine_worse_frac": float(np.mean(de > 0)),
            "translation_median": float(np.median(et)),
            "affine_median": float(np.median(ea)),
            "center_delta_median": float(np.median(center)),
            "center_delta_p90": float(np.percentile(center, 90)),
        })
    return out


def write_csv(path, rows):
    fields = [
        "threshold", "accepted", "accepted_frac",
        "median_delta", "mean_delta", "p90_delta", "affine_worse_frac",
        "translation_median", "affine_median",
        "center_delta_median", "center_delta_p90",
    ]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="affine_gate_sweep.csv")
    ap.add_argument("--plot", default="affine_gate_sweep.png")
    args = ap.parse_args()

    by_seq = {}
    all_rows = []
    for item in args.csvs:
        path = Path(item)
        rows = load_rows(path)
        by_seq[rows[0]["seq"] if rows else path.stem] = rows
        all_rows.extend(rows)

    cond = np.array([
        r["affine_h_cond"] for r in all_rows
        if np.isfinite(r["err_translation"])
        and np.isfinite(r["err_affine"])
        and np.isfinite(r["affine_h_cond"])
    ], float)
    percentiles = [5, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 95, 100]
    thresholds = np.unique(np.percentile(cond, percentiles))

    combined = summarize(all_rows, thresholds)
    write_csv(args.out, combined)
    print(f"saved {args.out}")

    print("\ncombined gate sweep:")
    print("pct  threshold    accept  med_delta  worse  trans_med  aff_med")
    for pct, row in zip(percentiles, combined):
        print(
            f"{pct:3d}  {row['threshold']:10.1f}  "
            f"{100*row['accepted_frac']:6.1f}%  "
            f"{row['median_delta']:+8.4f}  "
            f"{100*row['affine_worse_frac']:5.1f}%  "
            f"{row['translation_median']:.4f}  {row['affine_median']:.4f}"
        )

    for seq, rows in by_seq.items():
        seq_out = summarize(rows, thresholds)
        if not seq_out:
            continue
        best = min(seq_out, key=lambda r: r["median_delta"])
        print(
            f"{seq}: best median delta {best['median_delta']:+.4f}px at "
            f"cond<={best['threshold']:.1f}, accept={100*best['accepted_frac']:.1f}%, "
            f"affine_worse={100*best['affine_worse_frac']:.1f}%"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([100 * r["accepted_frac"] for r in combined])
    med = np.array([r["median_delta"] for r in combined])
    worse = np.array([100 * r["affine_worse_frac"] for r in combined])
    p90 = np.array([r["p90_delta"] for r in combined])
    cmed = np.array([r["center_delta_median"] for r in combined])

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(x, med, marker="o", label="median")
    ax[0].plot(x, p90, marker="o", label="p90")
    ax[0].axhline(0.0, color="k", lw=1)
    ax[0].set_xlabel("accepted pairs [%]")
    ax[0].set_ylabel("GT error delta affine-translation [px]")
    ax[0].legend()
    ax[1].plot(x, worse, marker="o")
    ax[1].axhline(50.0, color="k", lw=1)
    ax[1].set_xlabel("accepted pairs [%]")
    ax[1].set_ylabel("affine worse [%]")
    ax[2].plot(x, cmed, marker="o")
    ax[2].set_xlabel("accepted pairs [%]")
    ax[2].set_ylabel("median center delta [px]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130)
    print(f"saved {args.plot}")


if __name__ == "__main__":
    main()
