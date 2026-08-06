"""WHY is there a per-track lean? Two zero-free-parameter predictions.

We established the relative pose error within a track is a constant twist xi_a=(w_a,v_a)
times the window length. That is a DESCRIPTION. This tests two candidate CAUSES, each of
which predicts xi_a exactly, with no fitted constant:

  TRANSLATION -- the filter's velocity estimate is biased over stretches of time. A velocity
  error dv integrates to a position error dv*L*dt, so in the anchor camera frame

        v_a  ==  R_bc^T (v_est_body - v_gt_body) * dt                          (prediction P1)

  ROTATION -- a residual gyro-bias error db_g rotates the frame at a constant rate, so

        w_a  ==  R_bc^T (bg_est - bg_true) * dt                                 (prediction P2)

Note the exponent already EXCLUDES acceleration-level causes: a constant tilt or accel bias
corrupts acceleration, giving position error ~L^2, but the measured translation error grows
as L^1. So the lean must live at the rate/velocity level, which is what both predictions are.

P1 is testable on MidAir (vel_est_body/vel_gt_body are recorded). P2 needs true gyro bias,
which EuRoC publishes in state_groundtruth_estimate0 (columns b_w_RS_S_*).

A match means the cause is identified. A mismatch means the lean is something else -- most
likely filter-internal (linearisation / observability), which no sensor-level model explains.
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

FIT = list(range(4, 65, 8))
SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
      "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")


def se3(R, p):
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, p; return T


def twists(Te, Tg, A):
    numw = np.zeros((len(A), 3)); numv = np.zeros((len(A), 3)); den = 0.0
    for L in FIT:
        Xe = np.linalg.inv(Te[A]) @ Te[A + L]
        Xg = np.linalg.inv(Tg[A]) @ Tg[A + L]
        numw += L * Rot.from_matrix(np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]).as_rotvec()
        numv += L * (Xe[:, :3, 3] - Xg[:, :3, 3]); den += L * L
    return numw / den, numv / den


def score(pred, actual, label):
    """How well does a zero-parameter prediction match? ratio of medians + direction."""
    pn = np.linalg.norm(pred, axis=1); an = np.linalg.norm(actual, axis=1)
    ok = (pn > 0) & (an > 0)
    if ok.sum() < 20:
        return None
    cos = (pred[ok] * actual[ok]).sum(1) / (pn[ok] * an[ok])
    ratio = np.median(pn[ok] / an[ok])
    # fraction of the actual twist explained if we simply subtract the prediction
    resid = np.median(((actual[ok] - pred[ok]) ** 2).sum(1))
    tot = np.median((actual[ok] ** 2).sum(1))
    return dict(label=label, n=int(ok.sum()), ratio=ratio,
                cos=float(np.median(cos)), expl=1 - resid / tot)


def main():
    dt_mid, dt_euroc = 1.0 / 25.0, 1.0 / 20.0
    R_bc = md.RT_BC[:3, :3]

    print("P1  TRANSLATION = velocity error?   v_a  vs  R_bc^T (v_est - v_gt) dt   [MidAir]")
    print(f"{'run':<10}{'n':>7}{'|pred|/|actual|':>17}{'median cos':>12}{'energy expl':>13}")
    print("-" * 60)
    for i in range(3):
        f = f"{SP}/mono_run_fig_traj{i}.npz"
        if not Path(f).exists():
            continue
        d = np.load(f); p = "baseline_"
        ep, eq, gp, gq = d[p+"est"], d[p+"quat"], d[p+"gt"], d[p+"gt_quat"]
        ve, vg = d[p+"vel_est_body"], d[p+"vel_gt_body"]
        n = len(ep)
        Te = np.array([se3(Rot.from_quat(eq[j]).as_matrix(), ep[j]) @ md.RT_BC for j in range(n)])
        Tg = np.array([se3(Rot.from_quat(gq[j]).as_matrix(), gp[j]) @ md.RT_BC for j in range(n)])
        A = np.arange(200, n - 70, 11)
        _, v_a = twists(Te, Tg, A)
        # average velocity error over each window, mapped into the anchor camera frame
        Lm = int(np.mean(FIT))
        verr = np.array([(ve[a:a+Lm] - vg[a:a+Lm]).mean(0) for a in A])
        pred = (R_bc.T @ verr.T).T * dt_mid
        s = score(pred, v_a, f"VO t{i}")
        if s:
            print(f"{s['label']:<10}{s['n']:>7}{s['ratio']:>17.3f}{s['cos']:>12.3f}"
                  f"{100*s['expl']:>12.1f}%")

    print("\nP2  ROTATION = residual gyro bias?  w_a  vs  R_bc^T (bg_est - bg_true) dt   [EuRoC]")
    print(f"{'run':<18}{'n':>7}{'|pred|/|actual|':>17}{'median cos':>12}{'energy expl':>13}")
    print("-" * 68)
    seqs = [("V1_01_easy", "vicon_room1/V1_01_easy/V1_01_easy"),
            ("V1_02_medium", "vicon_room1/V1_02_medium/V1_02_medium"),
            ("V1_03_difficult", "vicon_room1/V1_03_difficult/V1_03_difficult"),
            ("V2_01_easy", "vicon_room2/V2_01_easy/V2_01_easy"),
            ("V2_02_medium", "vicon_room2/V2_02_medium/V2_02_medium"),
            ("V2_03_difficult", "vicon_room2/V2_03_difficult/V2_03_difficult")]
    for nm, rel in seqs:
        f = f"{SP}/bx_euroc_{nm}.npz"
        if not Path(f).exists():
            continue
        d = np.load(f); p = "baseline_"
        ep, eq, gp, gq = d[p+"est"], d[p+"quat"], d[p+"gt"], d[p+"gt_quat"]
        bge, tbc = d[p+"gyro_bias"], d[p+"t_bc"]
        # true gyro bias from EuRoC GT csv (cols 11-13), resampled to frame count
        path = Path("/mnt/18TB/chen_fu_yeh/datasets/EuRoC") / rel / "mav0" / \
            "state_groundtruth_estimate0" / "data.csv"
        bt = []
        with open(path) as fh:
            r = csv.reader(fh); next(r)
            for row in r:
                bt.append([float(row[11]), float(row[12]), float(row[13])])
        bt = np.array(bt)
        n = len(ep)
        idx = np.linspace(0, len(bt) - 1, n).astype(int)
        bgt = bt[idx]
        Te = np.array([se3(Rot.from_quat(eq[j]).as_matrix(), ep[j]) @ tbc for j in range(n)])
        Tg = np.array([se3(Rot.from_quat(gq[j]).as_matrix(), gp[j]) @ tbc for j in range(n)])
        A = np.arange(200, n - 70, 11)
        w_a, _ = twists(Te, Tg, A)
        Lm = int(np.mean(FIT))
        berr = np.array([(bge[a:a+Lm] - bgt[a:a+Lm]).mean(0) for a in A])
        pred = (tbc[:3, :3].T @ berr.T).T * dt_euroc
        s = score(pred, w_a, nm)
        if s:
            print(f"{s['label']:<18}{s['n']:>7}{s['ratio']:>17.3f}{s['cos']:>12.3f}"
                  f"{100*s['expl']:>12.1f}%")
    print("\nratio ~1 AND cos ~1 AND energy explained >>0  =>  cause identified")
    print("ratio far from 1 or cos ~0                    =>  not this; likely filter-internal")


if __name__ == "__main__":
    main()
