"""Publication trajectory figure (top-down XY over side-view XZ) from prior_ab /
midair_vio_run pose npz files — one column per trajectory.

READS each npz's `<npz>.manifest.json` (written by run_manifest.py), optionally
ASSERTS the config hash, and STAMPS the config fingerprint on the figure — so a
figure can never silently come from the wrong/modified config. See memory:
prevent-config-provenance-loss ("writing it is half — you must read it").

Usage:
  plot_trajectory_figure.py --npz t2.npz t0.npz t1.npz --out fig.png \
      [--mode-key auto] [--expect-hash dcf7b5e867500a0e] [--title "..."]
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def umeyama(src, dst):
    """SE(3) (no scale) src->dst; returns aligned src and RMSE."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (dst - mu_d).T @ (src - mu_s) / len(src)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; t = mu_d - R @ mu_s
    a = (R @ src.T).T + t
    return a, float(np.sqrt(((a - dst) ** 2).sum(1).mean()))


def mode_prefix(npz, override):
    if override and override != "auto":
        return override
    for k in npz.files:
        if k.endswith("_est"):
            return k[:-4]           # prior_ab mode-prefixed format
    if "est" in npz.files:
        return ""                    # midair_vio_run flat format
    raise SystemExit("no *_est or 'est' key in npz; pass --mode-key")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", required=True, help="pose npz files, one per column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode-key", default="auto", help="mode prefix (default auto-detect *_est)")
    ap.add_argument("--expect-hash", default="", help="assert every manifest config_sha256_16 == this")
    ap.add_argument("--title", default="full-trajectory VIO")
    args = ap.parse_args()

    n = len(args.npz)
    fig, ax = plt.subplots(2, n, figsize=(5.3 * n, 9), sharex="col", squeeze=False)
    metas = []; git = ""; mode = ""
    for j, path in enumerate(args.npz):
        d = np.load(path)
        pre = mode_prefix(d, args.mode_key)
        est, gt = (d[pre + "_est"], d[pre + "_gt"]) if pre else (d["est"], d["gt"])
        man = json.load(open(path + ".manifest.json"))
        kp = man["key_params"]
        if args.expect_hash:
            assert man["config_sha256_16"] == args.expect_hash, (
                f"{path}: config {man['config_sha256_16']} != expected {args.expect_hash}")
        esta, ate = umeyama(est, gt)
        tl = np.linalg.norm(np.diff(gt, axis=0), axis=1).sum()
        traj = man.get("extra", {}).get("traj", j)
        a0, a1 = ax[0, j], ax[1, j]
        a0.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="Ground truth", zorder=2)
        a0.plot(esta[:, 0], esta[:, 1], color="#d1495b", lw=1.3, label="VIO estimate", zorder=3)
        a0.plot(gt[0, 0], gt[0, 1], "o", color="#2e8b57", ms=8)
        a0.plot(gt[-1, 0], gt[-1, 1], "s", color="#1f6feb", ms=7)
        a0.set_title(f"trajectory_{int(traj):04d}   ATE {100 * ate / tl:.2f}%  ({ate:.1f} m)", fontsize=12)
        a0.set_ylabel("y [m]"); a0.set_aspect("equal", "box"); a0.grid(alpha=.3, lw=.5)
        zerr = np.sqrt(np.mean((esta[:, 2] - gt[:, 2]) ** 2))
        a1.plot(gt[:, 0], gt[:, 2], "k-", lw=1.8)
        a1.plot(esta[:, 0], esta[:, 2], color="#d1495b", lw=1.1)
        a1.set_title(f"side view  (z-RMSE {zerr:.1f} m)", fontsize=11)
        a1.set_xlabel("x [m]"); a1.set_ylabel("z (up) [m]"); a1.grid(alpha=.3, lw=.5)
        metas.append((int(traj), man["config_path"].split("/")[-1], man["config_sha256_16"]))
        git = man["git_rev"]; mode = man.get("extra", {}).get("modes", pre)
    ax[0, 0].legend(loc="best", fontsize=10, framealpha=.9)
    fig.suptitle(args.title, fontsize=13, y=0.995)
    if len({h for _, _, h in metas}) == 1:  # one config across all panels
        _, c, h = metas[0]
        footer = f"config: {c} #{h} | mode={mode} | git {git}"
    else:                                    # per-trajectory configs — list each
        footer = "configs: " + ", ".join(f"traj_{t:04d}={c}#{h[:8]}" for t, c, h in metas) + f" | mode={mode} | git {git}"
    fig.text(0.5, 0.004, footer, ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}  ({n} trajectories, verified hash={args.expect_hash or 'n/a'})")


if __name__ == "__main__":
    main()
