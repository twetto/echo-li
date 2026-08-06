"""Held-out validation: can the gyro-bias drift rate SET the coherent q_k coefficient?

The per-track error twist xi_a is what a coherent range-noise term needs (E[|xi_a|^2]), and
|dbg| -- the drift of the filter's own gyro-bias estimate over the window -- correlates with
|xi_a| at rho ~ 0.7-0.84 on three runs. A rank correlation is not a coefficient, so this fits
an actual predictor and validates it on runs it never saw:

    log |xi_a|  =  alpha + beta * log |dbg_a|          (fit per-anchor, pooled over TRAIN runs)

Leave-one-run-out. On each held-out run, compare against the honest baseline a shipped
constant would give -- the mean of TRAIN |xi| -- using median absolute log-error, i.e. the
typical multiplicative miss:

    err = median | log( predicted / actual ) |      (0 = perfect, 0.69 = typical factor of 2)

  |dbg| model beats constant on held-out  => the coefficient is computable online.
  it does not                             => the correlation was in-sample structure only,
                                             and no shipped constant OR |dbg| rule is founded.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import midair_drift as md  # noqa: E402

FIT = list(range(4, 65, 8))
L0 = 32
SP = ("/tmp/claude-1002/-mnt-18TB-chen-fu-yeh-Documents-repos-echo-li/"
      "6d85e9b6-9f98-49f5-ba40-7b63477c673e/scratchpad")


def se3(R, p):
    T = np.eye(4); T[:3, :3], T[:3, 3] = R, p; return T


def load(f):
    d = np.load(f)
    has = lambda k: k in d.files
    if has("baseline_pcov_pos"):
        p = "baseline_"
        ep, eq, gp, gq = d[p+"est"], d[p+"quat"], d[p+"gt"], d[p+"gt_quat"]
        bg, ba = d[p+"gyro_bias"], d[p+"accel_bias"]
        tbc = d[p+"t_bc"] if has(p+"t_bc") else md.RT_BC
    else:
        if not has("gyro_bias"):
            return None
        ep, eq, k = d["est"], d["quat"], d["k"]
        bg, ba = d["gyro_bias"], d["accel_bias"]
        gp, gq = d["gt"], None
        # midair_vio_run stores GT position only; recover attitude is not needed if
        # gt_quat absent -> skip such runs rather than guess a convention.
        return None if gq is None else None
    n = len(ep)
    Te = np.array([se3(Rot.from_quat(eq[i]).as_matrix(), ep[i]) @ tbc for i in range(n)])
    Tg = np.array([se3(Rot.from_quat(gq[i]).as_matrix(), gp[i]) @ tbc for i in range(n)])
    return Te, Tg, np.asarray(bg, float), np.asarray(ba, float), n


def features(f):
    r = load(f)
    if r is None:
        return None
    Te, Tg, bg, ba, n = r
    A = np.arange(200, n - 70, 11)
    if len(A) < 40:
        return None
    numw = np.zeros((len(A), 3)); numv = np.zeros((len(A), 3)); den = 0.0
    for L in FIT:
        Xe = np.linalg.inv(Te[A]) @ Te[A + L]
        Xg = np.linalg.inv(Tg[A]) @ Tg[A + L]
        numw += L * Rot.from_matrix(np.linalg.inv(Xg[:, :3, :3]) @ Xe[:, :3, :3]).as_rotvec()
        numv += L * (Xe[:, :3, 3] - Xg[:, :3, 3]); den += L * L
    w = np.linalg.norm(numw / den, axis=1)
    v = np.linalg.norm(numv / den, axis=1)
    dbg = np.array([np.linalg.norm(bg[a + L0] - bg[a]) for a in A])
    dba = np.array([np.linalg.norm(ba[a + L0] - ba[a]) for a in A])
    ok = (w > 0) & (v > 0) & (dbg > 0) & (dba > 0)
    return w[ok], v[ok], dbg[ok], dba[ok]


def main():
    cands = ([(f"VO t{i}", f"{SP}/mono_run_fig_traj{i}.npz") for i in range(3)]
             + [(f"EuRoC {s}", f"{SP}/bx_euroc_{s}.npz") for s in
                ["V1_01_easy", "V1_02_medium", "V1_03_difficult",
                 "V2_01_easy", "V2_02_medium", "V2_03_difficult"]]
             + [(f"TUM room{i}", f"{SP}/bx_tumvi_room{i}.npz") for i in range(1, 7)])
    data = {}
    for nm, f in cands:
        if not Path(f).exists():
            continue
        r = features(f)
        if r is not None:
            data[nm] = r
    print(f"runs usable: {len(data)}  ({', '.join(data)})\n")
    if len(data) < 4:
        print("too few runs for leave-one-out"); return

    for chan, idx in [("ROTATION", 0), ("TRANSLATION", 1)]:
        print(f"=== {chan}: leave-one-run-out, median |log(pred/actual)| "
              f"(lower better; 0.69 = factor 2)")
        print(f"{'held-out run':<16}{'constant':>10}{'|dbg| model':>13}{'|dba| model':>13}"
              f"{'best':>8}")
        print("-" * 62)
        agg = {"const": [], "dbg": [], "dba": []}
        for ho in data:
            tr = [k for k in data if k != ho]
            Y = np.concatenate([np.log(data[k][idx]) for k in tr])
            res = {}
            res["const"] = np.median(np.abs(Y.mean() - np.log(data[ho][idx])))
            for tag, j in [("dbg", 2), ("dba", 3)]:
                X = np.concatenate([np.log(data[k][j]) for k in tr])
                b, a = np.polyfit(X, Y, 1)
                pred = a + b * np.log(data[ho][j])
                res[tag] = np.median(np.abs(pred - np.log(data[ho][idx])))
            best = min(res, key=res.get)
            for k in agg:
                agg[k].append(res[k])
            print(f"{ho:<16}{res['const']:>10.3f}{res['dbg']:>13.3f}{res['dba']:>13.3f}{best:>8}")
        print(f"{'MEAN':<16}{np.mean(agg['const']):>10.3f}{np.mean(agg['dbg']):>13.3f}"
              f"{np.mean(agg['dba']):>13.3f}")
        w = sum(1 for i in range(len(agg['const'])) if agg['dbg'][i] < agg['const'][i])
        print(f"  |dbg| beats a constant on {w}/{len(agg['const'])} held-out runs\n")


if __name__ == "__main__":
    main()
