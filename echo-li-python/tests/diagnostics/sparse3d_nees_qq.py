"""QQ plot of Sparse3D depth NEES against chi2_1, plus the shape diagnostics.

A single quantile cannot distinguish WHY a filter is inconsistent. The SHAPE of the
QQ deviation can:

  * straight line, slope != 1      -> uniform covariance mis-scale (one scalar fixes it)
  * matches low, diverges high     -> heavy tail; the error is not Gaussian, so no
                                      scalar fixes it (this is the Student-t / aperture case)
  * kink at some quantile          -> a MIXTURE of two populations

Reads the npz written by `midair_e2e_depth_nees.py --save-npz`.

  PY=.venv/bin/python
  $PY .../midair_e2e_depth_nees.py ... --save-npz nees_t0.npz
  $PY .../sparse3d_nees_qq.py nees_t0.npz --label "Kite t0" --out qq_t0.png
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    n = len(args.npz)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.4), squeeze=False)

    for j, path in enumerate(args.npz):
        z = np.load(path, allow_pickle=True)
        if "data" in z and "cols" in z:
            cols = [str(c) for c in z["cols"]]
            v = np.asarray(z["data"], float)[:, cols.index("nees")]
        else:
            v = np.asarray(z["nees" if "nees" in z else list(z.keys())[0]], float)
        v = v[np.isfinite(v) & (v >= 0)]
        lab = args.label or path.split("/")[-1]

        # --- QQ vs chi2_1 -------------------------------------------------------
        m = len(v)
        pp = (np.arange(1, m + 1) - 0.5) / m
        theo = stats.chi2.ppf(pp, df=1)
        emp = np.sort(v)
        ax = axes[0][j]
        ax.loglog(theo, emp, ".", ms=1.6, alpha=0.45, color="#2b6cb0")
        lo = max(theo[0], 1e-6)
        hi = theo[-1] * 1.2
        ax.loglog([lo, hi], [lo, hi], "k--", lw=1.2, label="ideal (y = x)")
        for q, c in [(0.95, "#c53030"), (0.99, "#805ad5")]:
            ax.axvline(stats.chi2.ppf(q, df=1), color=c, lw=0.9, ls=":",
                       label=f"chi2_{int(q * 100)}")
        ax.set_xlabel("chi2_1 theoretical quantile")
        ax.set_ylabel("empirical NEES quantile")
        ax.set_title(f"{lab}  (n={m})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.25, which="both")

        # --- local slope: d log(emp) / d log(theo) ------------------------------
        # Slope 1 everywhere with an offset = pure mis-scale. Slope RISING with
        # quantile = the tail is heavier than chi2, i.e. not fixable by scaling.
        k = max(m // 60, 20)
        ls_x, ls_y = [], []
        for i in range(k, m - k, max(k // 3, 1)):
            dx = np.log(theo[i + k]) - np.log(theo[i - k])
            dy = np.log(emp[i + k] + 1e-300) - np.log(emp[i - k] + 1e-300)
            if dx > 0:
                ls_x.append(pp[i])
                ls_y.append(dy / dx)
        ax2 = axes[1][j]
        ax2.plot(ls_x, ls_y, "-", lw=1.4, color="#2b6cb0")
        ax2.axhline(1.0, color="k", ls="--", lw=1.0, label="slope 1 (chi2-shaped)")
        ax2.axvline(0.95, color="#c53030", lw=0.9, ls=":")
        ax2.set_xlabel("probability level")
        ax2.set_ylabel("local log-log slope")
        ax2.set_ylim(0, max(3.0, (max(ls_y) if ls_y else 1) * 1.1))
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.25)

        med = np.median(v)
        print(f"{lab}: n={m}  median {med:.3f} (ideal 0.455)  mean {v.mean():.2f} "
              f"(ideal 1.0)")
        print(f"   %>chi2_95 {100 * (v > 3.84).mean():.1f}%  "
              f"%>chi2_99 {100 * (v > 6.63).mean():.1f}%  (ideal 5% / 1%)")
        # Scalar-rescale check: divide by (median/0.455) and see if the tail follows.
        sc = med / 0.455
        w = v / sc
        print(f"   after rescaling by median-ratio {sc:.2f}: "
              f"%>chi2_95 {100 * (w > 3.84).mean():.1f}%  "
              f"%>chi2_99 {100 * (w > 6.63).mean():.1f}%")
        print("   -> if these are still >> 5%/1%, the tail is SHAPE, not scale.")

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=140)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
