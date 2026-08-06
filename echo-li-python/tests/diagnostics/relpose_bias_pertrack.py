"""Is the coherent pose error persistent WITHIN a track but incoherent ACROSS tracks?

relpose_bias_falsification.py fitted ONE bias vector per run and found it explains almost
nothing (mean exponent 1.30 -> 1.27). But that pools every anchor epoch together, and the
relative error over [a,k] depends on the anchor a. If the underlying error wanders slowly,
each track sees a bias that is persistent over ITS window while differing across anchors --
which a global fit averages to zero, reproducing exactly that null result.

That distinction is what matters operationally: Sparse3D anchors each landmark and needs the
relative covariance for that landmark's own window, so the per-track window is the unit that
ships. This fits one bias vector PER ANCHOR instead:

    dphi(a, a+L) ~ w_a * L        (w_a track-specific)

and, to stop a 3-parameter fit from manufacturing the collapse, fits w_a on EVEN lags only
and scores the residual exponent on HELD-OUT ODD lags.

  per-track residual exponent ~1  => coherent WITHIN a track; the global null was an
                                     aggregation artefact, and the coefficient a coherent
                                     q_k term needs is the ACROSS-anchor spread of w_a.
  per-track residual still >>1    => not a bias at any scope; the coherent growth is
                                     something else entirely.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

FIT_LAGS = list(range(4, 65, 8))     # 4,12,20,...,60   (fit)
TEST_LAGS = list(range(8, 65, 8))    # 8,16,24,...,64   (held out)


def se3(R, p):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def load(npz, kite, root):
    d = np.load(npz)
    if "baseline_pcov_pos" in d.files:
        ep, eq = d["baseline_est"], d["baseline_quat"]
        gp, gq = d["baseline_gt"], d["baseline_gt_quat"]
        tbc = d["baseline_t_bc"] if "baseline_t_bc" in d.files else md.RT_BC
    else:
        ep, eq, k = d["est"], d["quat"], d["k"]
        ds = md.MidAir(root, "Kite_training", "sunny", kite, 1.0)
        T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        gp = np.zeros_like(ep); gq = np.zeros((len(k), 4))
        for i, kk in enumerate(k):
            M = T @ ds.pose(int(kk))
            gp[i], gq[i] = M[:3, 3], Rot.from_matrix(M[:3, :3]).as_quat()
        tbc = md.RT_BC
    n = len(ep)
    Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ tbc for i in range(n)])
    Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ tbc for i in range(n)])
    return Te, Tg, n


def dphi(Te, Tg, a, L):
    Xe = np.linalg.inv(Te[a]) @ Te[a + L]
    Xg = np.linalg.inv(Tg[a]) @ Tg[a + L]
    return Rot.from_matrix(np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]).as_rotvec()


def run(Te, Tg, n, burn=200, stride=17):
    anchors = np.arange(burn, n - max(TEST_LAGS) - 1, stride)
    if len(anchors) < 30:
        return None
    # per-anchor fit on FIT_LAGS
    num = np.zeros((len(anchors), 3)); den = 0.0
    for L in FIT_LAGS:
        num += L * dphi(Te, Tg, anchors, L); den += L * L
    w_a = num / den                                    # per-anchor bias, rad/frame
    # global fit for contrast (what the previous test did)
    w_g = w_a.mean(0)
    out = {}
    for tag, w in [("raw", None), ("global", w_g), ("pertrack", w_a)]:
        ys = []
        for L in TEST_LAGS:                            # HELD-OUT lags only
            r = dphi(Te, Tg, anchors, L)
            if w is not None:
                r = r - (w * L if w.ndim == 1 else w * L)
            ys.append(np.median((r * r).sum(1)))
        out[tag] = (float(np.polyfit(np.log(TEST_LAGS), np.log(ys), 1)[0]), np.array(ys))
    # how much does the per-anchor bias vary across anchors, vs its own magnitude?
    spread = np.linalg.norm(w_a.std(0)) / max(np.linalg.norm(w_a.mean(0)), 1e-12)
    return out, spread, w_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/18TB/chen_fu_yeh/datasets/dataset_MidAir/MidAir")
    a = ap.parse_args()
    SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
          "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")
    runs = [("MidAir", "VO t0", f"{SP}/mono_run_fig_traj0.npz", None),
            ("MidAir", "VO t1", f"{SP}/mono_run_fig_traj1.npz", None),
            ("MidAir", "VO t2", f"{SP}/mono_run_fig_traj2.npz", None),
            ("MidAir", "Kite t0", f"{SP}/kite_traj0.npz", 0),
            ("MidAir", "Kite t1", f"{SP}/kite_traj1.npz", 1)]
    runs += [("EuRoC", s.split('_')[0] + "_" + s.split('_')[1], f"{SP}/euroc_{s}.npz", None)
             for s in ["V1_01", "V1_02_medium", "V1_03_difficult",
                       "V2_01_easy", "V2_02_medium", "V2_03_difficult"]]
    runs += [("TUM-VI", f"room{i}", f"{SP}/tumvi_room{i}.npz", None) for i in range(1, 7)]

    print("exponent on HELD-OUT lags after removing a bias fitted on disjoint lags\n")
    print(f"{'ds':<8}{'run':<14}{'raw':>7}{'global':>9}{'per-track':>11}{'drop(pt)':>10}"
          f"{'across-anchor spread':>22}")
    print("-" * 82)
    R = []
    for ds, name, f, kite in runs:
        if not Path(f).exists():
            continue
        Te, Tg, n = load(f, kite, a.root)
        res = run(Te, Tg, n)
        if res is None:
            continue
        out, spread, _ = res
        e0 = out["raw"][0]; eg = out["global"][0]; ep = out["pertrack"][0]
        R.append((e0, eg, ep))
        print(f"{ds:<8}{name:<14}{e0:>7.2f}{eg:>9.2f}{ep:>11.2f}{e0-ep:>10.2f}{spread:>21.1f}x")
    R = np.array(R)
    print(f"\nmean over {len(R)} runs:  raw {R[:,0].mean():.2f}   "
          f"global-fit {R[:,1].mean():.2f}   PER-TRACK {R[:,2].mean():.2f}")
    print("\nper-track ~1 => coherent within a track; the global null was an aggregation artefact.")


if __name__ == "__main__":
    main()
