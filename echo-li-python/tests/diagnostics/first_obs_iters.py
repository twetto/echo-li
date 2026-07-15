"""Why does first-observation (reference-template) KLT lose? -- down to every iteration.

On Mid-Air with EXACT GT (midair_drift.py), first-obs beta explodes at age 40+ (3.7->12 px)
while previous-frame stays bounded (1.0->2.0). This instrument traces the first-obs
Gauss-Newton solve for *failing* tracks at every pyramid level and every iteration, and
runs the decisive test:

  Does the GN CONVERGE (residual down, step->0, well-conditioned Hessian)?  and
  Is the SSD OBJECTIVE minimum AT the true GT location, or displaced?

  -> optimizer FAILS  = GN stalls/oscillates or objective min IS at GT but it never reaches it
  -> objective WRONG  = GN converges cleanly to a stable minimum that is NOT at GT
                        (the stale reference photometrically prefers a drifted spot)

The second is "staleness": no optimizer/iters/basin fixes it (result 9's oracle-affine
finding). We also log NCC(reference, current@GT) to quantify appearance drift, and contrast
against the previous-frame KLT that succeeds on the SAME track/frame.

  PY=echo-li-python/venv/bin/python
  $PY first_obs_iters.py --root ~/Server250/18TB/datasets/dataset_MidAir/MidAir \
      --cond sunny --traj 0 --scale 0.5 --frames 300 [--n-victims 6]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as LA

sys.path.insert(0, str(Path(__file__).resolve().parent))
import photometric_klt_ab as pk  # noqa: E402
import midair_drift as md  # noqa: E402  (MidAir, project_world, backproject_world, intrinsics)

LV, R, ITERS = md.LV, md.R, md.ITERS
OFF = np.arange(-R, R + 1, dtype=np.float32)
OFFX = np.repeat(OFF, len(OFF)); OFFY = np.tile(OFF, len(OFF))
LAM = 1e6  # blendB reference weight (== pure first-observation template)


def ssd_at(pyr_lv, c, tref_lv):
    """RMS SSD of the reference patch vs the current image sampled centred at c (level px)."""
    Iw = pk.sample(pyr_lv, np.array([c[0]]), np.array([c[1]]), OFFX, OFFY)[0]
    return float(np.sqrt(np.mean((Iw - tref_lv) ** 2)))


def ncc_at(pyr_lv, c, tref_lv):
    Iw = pk.sample(pyr_lv, np.array([c[0]]), np.array([c[1]]), OFFX, OFFY)[0]
    a = Iw - Iw.mean(); b = tref_lv - tref_lv.mean()
    d = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-9
    return float(np.sum(a * b) / d)


def trace_solve(cur_pyr, cgx, cgy, p0, tref, gt, log):
    """Replicates pk.klt_track's blendB(lam huge)==first-observation forward-additive solve
    for ONE feature, logging every level+iteration. Returns converged (u,v)."""
    u = p0.astype(np.float64).copy()
    for lv in reversed(range(LV)):
        s = 0.5 ** lv
        Cl, Gx, Gy = cur_pyr[lv], cgx[lv], cgy[lv]
        Pl = cur_pyr[lv]                                    # blendB prev_pyr == cur_pyr
        c0x, c0y = p0[0] * s, p0[1] * s
        Tprev = pk.sample(Pl, np.array([c0x]), np.array([c0y]), OFFX, OFFY)
        T = (Tprev + LAM * tref[lv][None, :]) / (1.0 + LAM)  # == tref (lam huge)
        ux, uy = u[0] * s, u[1] * s
        for it in range(ITERS):
            Iw = pk.sample(Cl, np.array([ux]), np.array([uy]), OFFX, OFFY)
            jx = pk.sample(Gx, np.array([ux]), np.array([uy]), OFFX, OFFY)
            jy = pk.sample(Gy, np.array([ux]), np.array([uy]), OFFX, OFFY)
            res = Iw - T
            Hxx = float(np.sum(jx * jx)); Hxy = float(np.sum(jx * jy)); Hyy = float(np.sum(jy * jy))
            bx = -float(np.sum(jx * res)); by = -float(np.sum(jy * res))
            reg = 1e-3 * (Hxx + Hyy + 1e-6)
            Hxx += reg; Hyy += reg
            det = Hxx * Hyy - Hxy * Hxy
            dx = (Hyy * bx - Hxy * by) / det if abs(det) > 1e-6 else 0.0
            dy = (Hxx * by - Hxy * bx) / det if abs(det) > 1e-6 else 0.0
            step = np.hypot(dx, dy)
            scl = 1.0 / step if step > 1.0 else 1.0
            ux += dx * scl; uy += dy * scl
            tr = Hxx + Hyy; dt = Hxx * Hyy - Hxy * Hxy
            lam_min = 0.5 * (tr - np.sqrt(max(tr * tr - 4 * dt, 0.0)))
            lam_max = 0.5 * (tr + np.sqrt(max(tr * tr - 4 * dt, 0.0)))
            cond = lam_max / max(lam_min, 1e-9)
            absu = np.array([ux / s, uy / s])
            log.append(dict(lv=lv, it=it, u=absu.copy(),
                            rms=float(np.sqrt(np.mean(res ** 2))),
                            step=float(step * scl / s),
                            lam_min=lam_min, cond=cond,
                            dgt=float(np.hypot(*(absu - gt)))))
        u[0], u[1] = ux / s, uy / s
    return u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--set", dest="subset", default="VO_test")
    ap.add_argument("--cond", default="sunny")
    ap.add_argument("--traj", type=int, default=0)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--redetect", type=int, default=250)
    ap.add_argument("--capture-ages", default="40,80")
    ap.add_argument("--first-thr", type=float, default=5.0, help="first-obs beta to call a loss")
    ap.add_argument("--prev-thr", type=float, default=1.5, help="prev beta below which prev is fine")
    ap.add_argument("--n-victims", type=int, default=6)
    ap.add_argument("--out", default="first_obs_iters")
    args = ap.parse_args()
    cap_ages = set(int(a) for a in args.capture_ages.split(","))

    ds = md.MidAir(args.root, args.subset, args.cond, args.traj, args.scale)
    W, H = ds.image(0).shape[1], ds.image(0).shape[0]
    f, cx, cy = md.intrinsics(W, H)
    print(f"Mid-Air {args.subset}/{args.cond}/{ds.traj}  work {W}x{H} f={f:.0f}")

    Xw = {}; born = {}; first_patch = {}; pos_prev = {}; pos_first = {}
    nid = 0
    victims = []
    prev = None
    for i in range(min(args.frames, ds.n)):
        img = ds.image(i); depth = ds.depth(i); T = ds.pose(i)
        cur_pyr, cgx, cgy = pk.build_pyramid(img, LV, histeq=False)
        init_first = {j: pos_first[j].copy() for j in pos_first}     # pre-update (trace init)
        if prev is not None:
            ppyr, pgx, pgy = prev
            idp = list(pos_prev)
            if idp:
                u, v, _ = pk.klt_track(ppyr, cur_pyr, cgx, cgy,
                                       np.array([pos_prev[j] for j in idp]), R, ITERS, "ssd")
                for k, j in enumerate(idp):
                    if v[k]:
                        pos_prev[j] = u[k]
                    else:
                        pos_prev.pop(j, None)
            idf = list(pos_first)
            if idf:
                tref = np.array([first_patch[j] for j in idf])
                u, v, _ = pk.klt_track(cur_pyr, cur_pyr, cgx, cgy,
                                       np.array([pos_first[j] for j in idf]), R, ITERS,
                                       "blendB", "ssd", LAM, tref)
                for k, j in enumerate(idf):
                    if v[k]:
                        pos_first[j] = u[k]
                    else:
                        pos_first.pop(j, None)
            for j in list(born):
                gp, rng, _z = md.project_world(Xw[j], T, f, cx, cy)
                inb = gp is not None and R + 2 <= gp[0] < W - R - 2 and R + 2 <= gp[1] < H - R - 2
                if not inb or (j not in pos_prev and j not in pos_first):
                    for d in (Xw, born, first_patch, pos_prev, pos_first):
                        d.pop(j, None)
                    continue
                age = i - born[j]
                if (len(victims) < args.n_victims and age in cap_ages
                        and j in pos_first and j in pos_prev):
                    bf = np.hypot(*(pos_first[j] - gp)); bp = np.hypot(*(pos_prev[j] - gp))
                    if bf > args.first_thr and bp < args.prev_thr:
                        log = []
                        conv = trace_solve(cur_pyr, cgx, cgy, init_first[j],
                                           first_patch[j], gp, log)
                        # objective landscape around GT (finest level, +/-8 px)
                        gx, gy = gp
                        grid = np.arange(-8, 8.01, 0.5)
                        land = np.array([[ssd_at(cur_pyr[0], (gx + dx, gy + dy), first_patch[j][0])
                                          for dx in grid] for dy in grid])
                        aij = np.unravel_index(np.argmin(land), land.shape)
                        argmin_off = np.array([grid[aij[1]], grid[aij[0]]])
                        victims.append(dict(
                            id=j, frame=i, age=age, gt=gp.copy(),
                            init=init_first[j].copy(), conv=conv.copy(),
                            pos_prev=pos_prev[j].copy(), bf=bf, bp=bp, log=log,
                            land=land, grid=grid, argmin_off=argmin_off,
                            ncc_gt=ncc_at(cur_pyr[0], (gx, gy), first_patch[j][0]),
                            ncc_conv=ncc_at(cur_pyr[0], tuple(conv), first_patch[j][0]),
                            ssd_gt=ssd_at(cur_pyr[0], (gx, gy), first_patch[j][0]),
                            ssd_conv=ssd_at(cur_pyr[0], tuple(conv), first_patch[j][0]),
                            crop=cv2.getRectSubPix(img, (48, 48), (float(gx), float(gy))),
                            ref0=first_patch[j][0].reshape(2 * R + 1, 2 * R + 1).T,
                            cur0=pk.sample(cur_pyr[0], np.array([gx]), np.array([gy]),
                                           OFFX, OFFY)[0].reshape(2 * R + 1, 2 * R + 1).T))
        # births
        if len(pos_prev) < args.redetect:
            mask = np.uint8((depth < md.SKY) & (depth > 1.0)) * 255
            for j in pos_prev:
                p = pos_prev[j]; cv2.circle(mask, (int(p[0]), int(p[1])), R + 2, 0, -1)
            corners = cv2.goodFeaturesToTrack(img, args.max_tracks - len(pos_prev),
                                              0.01, 2 * R + 3, mask=mask)
            if corners is not None:
                for c in corners.reshape(-1, 2):
                    x, y = float(c[0]), float(c[1]); gx, gy = int(round(x)), int(round(y))
                    d_rng = float(depth[gy, gx])
                    if not (1.0 < d_rng < md.SKY):
                        continue
                    Xw[nid] = md.backproject_world((x, y), d_rng, T, f, cx, cy)
                    born[nid] = i
                    first_patch[nid] = np.stack(
                        [pk.sample(cur_pyr[lv], np.array([x * 0.5 ** lv]),
                                   np.array([y * 0.5 ** lv]), OFFX, OFFY)[0] for lv in range(LV)])
                    pos_prev[nid] = np.array([x, y]); pos_first[nid] = np.array([x, y])
                    nid += 1
        prev = (cur_pyr, cgx, cgy)
        if len(victims) >= args.n_victims:
            break

    print(f"\ncaptured {len(victims)} first-obs-loses tracks (first>{args.first_thr}, "
          f"prev<{args.prev_thr})\n")
    report(victims, args.out)


def report(victims, out):
    for v in victims:
        gt = v["gt"]
        print("=" * 78)
        print(f"id {v['id']}  frame {v['frame']}  age {v['age']}   "
              f"first-obs beta {v['bf']:.2f}px   previous-frame beta {v['bp']:.2f}px")
        print(f"  init {v['init']}  ->  first-obs converged {v['conv']}   GT {gt}")
        print(f"  {'lv':>2} {'it':>3} {'pos_x':>8} {'pos_y':>8} {'rms':>7} {'step':>7} "
              f"{'lam_min':>8} {'cond':>8} {'dist_GT':>7}")
        for e in v["log"]:
            print(f"  {e['lv']:>2} {e['it']:>3} {e['u'][0]:8.2f} {e['u'][1]:8.2f} "
                  f"{e['rms']:7.2f} {e['step']:7.3f} {e['lam_min']:8.1f} {e['cond']:8.1f} "
                  f"{e['dgt']:7.2f}")
        conv_dgt = np.hypot(*(v["conv"] - gt))
        amin_dgt = np.hypot(*v["argmin_off"])
        last = v["log"][-1]
        conv_ok = last["step"] < 0.05
        obj_at_gt = amin_dgt < 1.0
        print(f"  --> GN {'CONVERGED' if conv_ok else 'did NOT settle'} "
              f"(final step {last['step']:.3f}px, cond {last['cond']:.0f})")
        print(f"  --> SSD-objective min is {amin_dgt:.2f}px from GT "
              f"(offset {v['argmin_off']}); converged is {conv_dgt:.2f}px from GT")
        print(f"  --> RMS: ref-vs-cur@GT {v['ssd_gt']:.1f}  @converged {v['ssd_conv']:.1f}   "
              f"NCC: @GT {v['ncc_gt']:.3f}  @converged {v['ncc_conv']:.3f}")
        if conv_ok and not obj_at_gt:
            print("  ==> OBJECTIVE WRONG: optimizer found a clean minimum that is NOT at GT "
                  "(stale reference) -- no optimizer/iters fix this.")
        elif not conv_ok:
            print("  ==> OPTIMIZER: GN never settled (basin/conditioning).")
        else:
            print("  ==> objective min AT GT but converged elsewhere -> init/basin issue.")
        print()

    # aggregate verdict
    if victims:
        objwrong = sum(1 for v in victims
                       if v["log"][-1]["step"] < 0.05 and np.hypot(*v["argmin_off"]) >= 1.0)
        print(f"VERDICT: {objwrong}/{len(victims)} losses = clean GN convergence to a "
              f"NON-GT objective minimum (stale-reference / appearance drift).")
        med_ncc = np.median([v["ncc_gt"] for v in victims])
        print(f"  median NCC(reference, current@GT) = {med_ncc:.3f} "
              f"(1.0 = identical appearance; low = reference no longer matches truth).")

    # figure: SSD landscape + GN trajectory per victim
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(victims)
    if n == 0:
        return
    fig, ax = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    if n == 1:
        ax = ax.reshape(2, 1)
    for c, v in enumerate(victims):
        g = v["grid"]; ext = [g[0], g[-1], g[-1], g[0]]
        a = ax[0, c]
        im = a.imshow(v["land"], extent=ext, origin="upper", cmap="viridis")
        traj = np.array([e["u"] - v["gt"] for e in v["log"] if e["lv"] == 0])
        if len(traj):
            a.plot(traj[:, 0], traj[:, 1], "-o", color="red", ms=2, lw=0.8, label="GN (L0)")
        a.plot(0, 0, "+", color="lime", ms=12, mew=2, label="GT")
        a.plot(*(v["conv"] - v["gt"]), "o", mfc="none", mec="red", ms=10, mew=1.5, label="converged")
        a.plot(*(v["pos_prev"] - v["gt"]), "x", color="gold", ms=8, mew=2, label="prev-KLT")
        a.plot(*v["argmin_off"], "s", mfc="none", mec="white", ms=9, label="SSD argmin")
        a.set_title(f"id{v['id']} age{v['age']}\n1st {v['bf']:.1f} / prev {v['bp']:.1f}px", fontsize=7)
        a.set_xlabel("px from GT", fontsize=6); a.tick_params(labelsize=6)
        if c == 0:
            a.legend(fontsize=5, loc="upper right")
        a2 = ax[1, c]
        rms = [e["rms"] for e in v["log"]]; dgt = [e["dgt"] for e in v["log"]]
        a2.plot(rms, color="steelblue", label="RMS residual")
        a2b = a2.twinx(); a2b.plot(dgt, color="crimson", label="dist to GT")
        a2.set_xlabel("iteration (all levels)", fontsize=6)
        a2.set_ylabel("RMS", fontsize=6, color="steelblue"); a2b.set_ylabel("dist GT px", fontsize=6, color="crimson")
        a2.tick_params(labelsize=6); a2b.tick_params(labelsize=6)
    fig.suptitle("first-obs GN solve on losing tracks: SSD landscape + trajectory (top), "
                 "residual & dist-to-GT per iter (bottom)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{out}_{victims[0]['id']}.png", dpi=140)
    print(f"\nsaved {out}_{victims[0]['id']}.png")


if __name__ == "__main__":
    main()
