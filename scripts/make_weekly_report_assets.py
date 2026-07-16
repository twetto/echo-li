#!/usr/bin/env python3
"""Generate weekly-report tables/figures from existing Sparse3D diagnostics.

This script intentionally does not run new experiments. It parses existing logs,
checks generated artifacts, creates Matplotlib summary figures, and writes a
small Markdown report suitable for advisor discussion.

Run from the echo-li repo root:

  echo-li-python/venv/bin/python scripts/make_weekly_report_assets.py \
    --log log_bias_suite.txt \
    --video-dir /tmp/midair_track_bias_suite_20260717_011357
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_suite_log(path: Path):
    text = path.read_text()
    text = text.split("Quick summary:", 1)[0]
    parts = re.split(r"^===== (VO_test_.*?) =====\n", text, flags=re.M)
    blocks = {}
    for i in range(1, len(parts), 2):
        name = parts[i]
        body = parts[i + 1]
        if name not in blocks and ("Mid-Air" in body or "temporal-correlation input" in body):
            blocks[name] = body

    temporal = []
    for name, body in blocks.items():
        if not name.endswith("_temporal"):
            continue
        sl = name.replace("VO_test_", "").replace("_temporal", "").replace("_traj", "/traj")
        m = re.search(
            r"temporal-correlation input \((\d+) pairs\).*?"
            r"track count usable/all: ([^\n]+).*?"
            r"radial \|error\| median/p90/p99: ([^\n]+).*?"
            r"outlier >3px radial: ([^\n]+)",
            body,
            re.S,
        )
        if not m:
            continue
        residual = [float(x) for x in re.findall(r"[0-9.]+", m.group(3))]
        clean = body.split("=== clean-core contiguous runs", 1)[1]
        taus = []
        neffs = []
        for comp in ["ex", "ey", "e_weak", "e_strong"]:
            cm = re.search(
                rf"\n{comp}:.*?tau_int raw ([0-9.]+).*?N_eff/N ~= ([0-9.]+)",
                clean,
                re.S,
            )
            if cm:
                taus.append(float(cm.group(1)))
                neffs.append(float(cm.group(2)))
        temporal.append(
            {
                "slice": sl,
                "pairs": int(m.group(1)),
                "usable_tracks": m.group(2).strip(),
                "res_med": residual[0],
                "res_p90": residual[1],
                "res_p99": residual[2],
                "outlier": float(m.group(4).replace("%", "")),
                "tau_min": min(taus),
                "tau_max": max(taus),
                "neff_min": min(neffs),
                "neff_max": max(neffs),
            }
        )

    bias = []
    for name, body in blocks.items():
        if "_bias_" not in name:
            continue
        sl = name.rsplit("_bias_", 1)[0].replace("VO_test_", "").replace("_traj", "/traj")
        sigma = float(re.search(r"\(bias sigma\s*([0-9.Ee+-]+)px\)", body).group(1))
        rep_all = float(re.search(r"3D NEES mean/median:\s*[0-9.Ee+-]+ /\s*([0-9.Ee+-]+)", body).group(1))
        iid_all = float(re.search(r"iid Fisher 3D mean/med:\s*[0-9.Ee+-]+ /\s*([0-9.Ee+-]+)", body).group(1))
        bias_all = float(re.search(r"bias Fisher 3D mean/med:\s*[0-9.Ee+-]+ /\s*([0-9.Ee+-]+)", body).group(1))
        clean_rep = float(re.search(r"valid full3\s+drift<=3px: n=\s*\d+\s+median\s*([0-9.Ee+-]+)", body).group(1))
        clean_iid = float(re.search(r"valid iid3\s+drift<=3px: n=\s*\d+\s+median\s*([0-9.Ee+-]+)", body).group(1))
        cm = re.search(
            r"valid bias3\s+drift<=3px: n=\s*(\d+)\s+median\s*([0-9.Ee+-]+)"
            r"\s+mean\s*([0-9.Ee+-]+)\s+p90\s*([0-9.Ee+-]+)",
            body,
        )
        drift = re.search(r"measurement drift vs GT:\s*median\s*([0-9.Ee+-]+)px\s*p90\s*([0-9.Ee+-]+)px", body)
        bias.append(
            {
                "slice": sl,
                "sigma": sigma,
                "rep_all": rep_all,
                "iid_all": iid_all,
                "bias_all": bias_all,
                "clean_rep": clean_rep,
                "clean_iid": clean_iid,
                "clean_bias": float(cm.group(2)),
                "clean_mean": float(cm.group(3)),
                "clean_p90": float(cm.group(4)),
                "clean_n": int(cm.group(1)),
                "drift_med": float(drift.group(1)),
                "drift_p90": float(drift.group(2)),
            }
        )
    return temporal, bias


def write_figures(out_dir: Path, temporal, bias):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    slices = ["sunny/traj0", "foggy/traj1000", "sunset/traj2000"]
    colors = {"sunny/traj0": "#2563eb", "foggy/traj1000": "#64748b", "sunset/traj2000": "#dc2626"}
    markers = {"sunny/traj0": "o", "foggy/traj1000": "s", "sunset/traj2000": "^"}

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for sl in slices:
        rr = sorted([r for r in bias if r["slice"] == sl], key=lambda x: x["sigma"])
        ax.plot([r["sigma"] for r in rr], [r["clean_bias"] for r in rr],
                marker=markers[sl], lw=2, color=colors[sl], label=sl)
        ax.axhline(rr[0]["clean_rep"], color=colors[sl], ls=":", alpha=0.35, lw=1)
    ax.axhline(3.0, color="black", ls="--", lw=1.2, label="ideal 3D NEES = 3")
    ax.set_xlabel(r"marginalized per-track bias sigma $\sigma_b$ [px]")
    ax.set_ylabel("clean-domain full 3D NEES median")
    ax.set_title("Same-track bearing-bias covariance fixes typical tangent overconfidence")
    ax.set_ylim(0, 42)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "track_bias_nees_sweep.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(slices))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    best = [next(t for t in temporal if t["slice"] == sl)["neff_max"] * 100 for sl in slices]
    worst = [next(t for t in temporal if t["slice"] == sl)["neff_min"] * 100 for sl in slices]
    ax.bar(x, best, color=[colors[sl] for sl in slices], alpha=0.35, label="best component")
    ax.bar(x, worst, color=[colors[sl] for sl in slices], alpha=0.95, label="worst component")
    ax.set_xticks(x, slices, rotation=15, ha="right")
    ax.set_ylabel(r"clean-core effective sample fraction $N_{eff}/N$ [%]")
    ax.set_title("Repeated KLT measurements are strongly correlated within a track")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_neff_by_condition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    width = 0.35
    p90 = [next(t for t in temporal if t["slice"] == sl)["res_p90"] for sl in slices]
    outlier = [next(t for t in temporal if t["slice"] == sl)["outlier"] for sl in slices]
    ax.bar(x - width / 2, p90, width, color="#0f766e", label="p90 two-frame residual [px]")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, outlier, width, color="#b45309", label="radial >3 px outliers [%]")
    ax.set_xticks(x, slices, rotation=15, ha="right")
    ax.set_ylabel("p90 residual [px]")
    ax2.set_ylabel("outlier rate [%]")
    ax.set_title("Foggy is a different validity/outlier regime")
    ax.grid(axis="y", alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "flow_residual_outlier_by_condition.png", dpi=180)
    plt.close(fig)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_report(report_path: Path, fig_dir: Path, temporal, bias, video_dir: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel_fig = fig_dir.relative_to(report_path.parent)

    slices = ["sunny/traj0", "foggy/traj1000", "sunset/traj2000"]
    temporal_rows = []
    for sl in slices:
        t = next(x for x in temporal if x["slice"] == sl)
        temporal_rows.append([
            sl,
            f"{t['res_med']:.3f} / {t['res_p90']:.3f} / {t['res_p99']:.3f}",
            f"{t['outlier']:.2f}%",
            f"{t['tau_min']:.1f}-{t['tau_max']:.1f}",
            f"{100*t['neff_min']:.1f}-{100*t['neff_max']:.1f}%",
        ])

    bias_rows = []
    for sl in slices:
        rr = {r["sigma"]: r for r in bias if r["slice"] == sl}
        bias_rows.append([
            sl,
            f"{rr[0.0]['clean_rep']:.2f}",
            f"{rr[0.0]['clean_iid']:.2f}",
            f"{rr[0.25]['clean_bias']:.2f}",
            f"{rr[0.5]['clean_bias']:.2f}",
            f"{rr[1.0]['clean_bias']:.2f}",
        ])

    videos = []
    if video_dir and video_dir.exists():
        for p in sorted(video_dir.glob("*measurements.mp4")):
            videos.append([p.name, str(p), f"{p.stat().st_size / (1024 * 1024):.1f} MB"])
    if not videos:
        videos.append(["not found", "Run `JOBS=3 RUN_VIDEO=1 RUN_TEMPORAL=0 BIAS_SIGMAS=0 bash scripts/run_midair_track_bias_suite.sh`", ""])

    existing_artifacts = [
        ["MidAir exact-vs-Rudolf videos", "Useful", "Shows correspondence-validity failures; use the three suite videos."],
        ["first_obs_* patch/landscape PNGs", "Optional appendix", "Good for tracker-template/forward-scene failure explanation, not central to weekly summary."],
        ["pose_noise_nees.png", "Background only", "Supports old calibrated simulation / pose-noise harness story."],
        ["/tmp/midair_rudolf_flow_model_sunny.png", "Background", "Good if discussing earlier scalar white+bias model, superseded by track-bias covariance."],
    ]

    text = f"""# Weekly Report: Sparse3D Measurement Consistency and Track-Bias Model

## One-Sentence Update

Since the calibrated simulation and EuRoC bias-drift discovery, the main result is that real Sparse3D inconsistency is not a scalar pixel-noise problem: valid same-surface KLT tracks need a same-track bearing/pixel bias covariance, while the remaining tail is dominated by correspondence-validity failures such as occlusion, foreground swaps, sky/no-return boundaries, borders, and depth edges.

## Progress Since Calibrated Simulation

1. **Calibrated simulation established the baseline.** The Rust Sparse3D math can be statistically sane under clean analytic Gaussian pixel measurements and controlled pose-noise injections. The old simulations also showed limits of 1D Gaussian-Beta depth filtering at long range; it is not the right main path for convergence or consistency.

2. **EuRoC exposed a real-measurement gap.** With GT poses and Leica pointcloud depth, Sparse3D had good median depth error (~3.7%) but huge depth NEES and near-reasonable NIS. Batch-vs-sequential on the same tracks showed the issue followed the measurements, not just sequential covariance collapse.

3. **MidAir exact GT removed EuRoC ground-truth ambiguity.** The EuRoC drift magnitude was partly contaminated by real GT/depth-edge issues. Exact-GT MidAir showed a smaller clean two-frame Rudolf-V core, but Sparse3D still over-counted repeated same-track measurements.

4. **3D decomposition located the failure.** Scalar depth NEES understated the problem. Full 3D NEES was dominated by tangent/image-plane covariance collapse, not radial depth alone.

5. **Texture anisotropy is real but insufficient.** Structure tensor direction matters; weak texture direction has higher flow residual scale. However, the two-frame inlier core is only about 0.05-0.24 px, too small to explain the effective consistency sigma needed by Sparse3D.

6. **Temporal correlation closes the gap.** Clean-core same-track residuals have long raw autocorrelation. Within-track demeaning removes most of it, identifying a persistent per-track correspondence offset/bias rather than ordinary colored jitter.

7. **Marginalized track-bias covariance works.** The model

```text
u_k = pi(T_k, X) + b_track + eps_k
eps_k ~ N(0, sigma_px^2 I)
b_track ~ N(0, sigma_b^2 I)
```

gives same-track off-diagonal covariance after marginalizing `b_track`. On sunny/sunset MidAir, `sigma_b ~= 0.5 px` brings clean-domain full 3D NEES median close to ideal.

## Key Tables

### Two-Frame Residual and Temporal Correlation

{md_table(['slice', 'residual med/p90/p99 [px]', '>3px outliers', 'clean tau_int range', 'clean N_eff/N range'], temporal_rows)}

### Clean-Domain Full 3D NEES Median

{md_table(['slice', 'Sparse3D reported', 'iid Fisher', 'bias 0.25px', 'bias 0.50px', 'bias 1.00px'], bias_rows)}

## Figures

![Track-bias NEES sweep]({rel_fig}/track_bias_nees_sweep.png)

![Temporal effective sample fraction]({rel_fig}/temporal_neff_by_condition.png)

![Residual and outlier severity by condition]({rel_fig}/flow_residual_outlier_by_condition.png)

## Videos

{md_table(['video', 'path', 'size'], videos)}

## Interpretation

- **Inlier model:** same-surface tracks should use a same-track bearing/pixel nuisance bias or equivalent information cap. This is a measurement covariance model, not a landmark process model.
- **Texture model:** structure-tensor anisotropy is useful for per-feature covariance and gating, but it does not explain repeated-update overconfidence alone.
- **Outlier/validity model:** foggy is a frontend/visibility failure regime. It sees mostly closer objects, so tree occlusion and foreground/background swaps dominate. Sunny/sunset are cleaner in median, but rare sky/no-return points occluded by mountain/terrain boundaries dominate the NEES mean.
- **Paper claim:** the transferable claim is the model form, not a universal `sigma_b`. The scale and outlier rate must be calibrated by condition/quality.

## Artifact Assessment

{md_table(['artifact', 'use', 'comment'], existing_artifacts)}

## More Plots Needed?

The three generated plots above are enough for a concise weekly report. For a paper-quality version, add two more:

1. **Video frame montage:** three stills from sunny/foggy/sunset measurement videos with exact GT and Rudolf-V markers. This would visually support the occlusion/foreground-swap and sky/mountain-boundary interpretation.
2. **Tail mass plot:** top 1/5/10% NEES mass by condition. This would make clear why medians are fixed by the bias covariance while means remain dominated by invalid correspondences.

"""
    report_path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="log_bias_suite.txt")
    ap.add_argument("--notes-dir", default="../ECHO-LI-notes/docs")
    ap.add_argument("--video-dir", default="/tmp/midair_track_bias_suite_20260717_011357")
    ap.add_argument("--report-name", default="weekly_sparse3d_measurement_report_2026_07_17.md")
    args = ap.parse_args()

    notes = Path(args.notes_dir)
    fig_dir = notes / "figures" / "weekly_2026_07_17"
    report_path = notes / args.report_name
    temporal, bias = parse_suite_log(Path(args.log))
    write_figures(fig_dir, temporal, bias)
    write_report(report_path, fig_dir, temporal, bias, Path(args.video_dir))
    print(f"wrote {report_path}")
    print(f"wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
