"""Side-by-side video: Sparse3D fed by Rudolf-V KLT tracks vs by FlowDep-F patch tracks.

Same TartanAir frames, same ground-truth camera pose for both the trackers and the two
Sparse3D filters (zero pose covariance), same Sparse3D settings. Only CONVERGED landmarks
(`query_range > 0`) are drawn, because only those reach a consumer.

Layout (each panel = the scaled frame):
  top row     KLT converged range | FlowDep-F converged range | DENSIFIED range | GT range
              (TartanAir: planar z x ray norm); one log colour scale
  bottom row  KLT signed error | FlowDep-F signed error | densified signed error | densified
              per-pixel sigma (log-range, %) with the running statistics
Densified range = the rule of fdf_densify_study.py (design note row 5b): each covered pixel
takes the NEAREST converged patch's Sparse3D log-range (footprint 8 px at the patch's pyramid
level, so 8 or 16 px at level 0); its variance = that patch's own + the disagreement of the
covering patches + a footprint term (tan(slant) * d / f)^2 for the distance d to the patch
centre + the scale term fully correlated across patches (eq. 35 structure). No 1/sum-of-weights
gain anywhere; pixels no patch covers stay unknown. Per-pixel error, coverage and NEES vs GT
are accumulated in the statistics panel.
Error colour: blue = estimate short, red = long, saturating at +-20 %; crosses = |err| > 20 %.

Usage:
  $PY $S --root ~/18TB/datasets/tartanair_v2/OldScandinavia/Data_easy --traj P000 \
      --config configs/diagnostics_tartanair_e2e_depth_nees.yaml --start 610 --frames 110 --out fdf_vs_klt_610.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import echo_li  # noqa: E402
from tartanair_e2e_depth_nees import RT_BC, SKY, SPARSE_KEYS, TartanAir, intrinsics  # noqa: E402
import midair_drift as md  # noqa: E402
import fdf_densify_study as dz  # noqa: E402


def txt(img, s, org, col=(235, 235, 235), scale=0.42):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("tartanair", "midair"), default="tartanair")
    ap.add_argument("--root", required=True)
    ap.add_argument("--traj", default="P000", help="TartanAir: P000; MidAir: trajectory number")
    ap.add_argument("--set", dest="subset", default="Kite_training", help="MidAir subset")
    ap.add_argument("--cond", default="sunny", help="MidAir condition")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", type=int, default=610)
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--range-lo", type=float, default=2.0)
    ap.add_argument("--range-hi", type=float, default=60.0)
    ap.add_argument("--max-pool", type=int, default=4000)
    ap.add_argument("--png-every", type=int, default=0, help="also write a PNG every N frames")
    ap.add_argument("--slant-deg", type=float, default=35.0, help="footprint slant prior of the densify rule")
    ap.add_argument("--dense-rule", default="nearest_disagree_slant", help="fusion rule name from fdf_densify_study")
    ap.add_argument("--fill", action="store_true", help="add the honest fill (normalised convolution, balance-gated) between measured patches; Filled pixels drawn paler")
    ap.add_argument("--sparse", action="append", default=[], metavar="KEY=VAL", help="SparseVog override")
    ap.add_argument("--fdf", action="append", default=[], metavar="KEY=VAL", help="FlowDepF override")
    args = ap.parse_args()

    T_nwu = np.eye(4)
    T_nwu[:3, :3] = np.diag([1.0, -1.0, -1.0])
    if args.dataset == "midair":
        ds = md.MidAir(args.root, args.subset, args.cond, int(args.traj), args.scale)
        im0 = ds.image(args.start)
        H, W = im0.shape
        f, cx, cy = md.intrinsics(W, H)
        ext = md.RT_BC
        sky = md.SKY
        n_cam = ds.n
        stamp_of = lambda k: k / 25.0  # noqa: E731
        gt_range_working = lambda k: ds.depth(k)  # already euclidean range at the working scale  # noqa: E731
    else:
        ds = TartanAir(args.root, args.traj, args.scale)
        im0 = ds.image(args.start)
        H, W = im0.shape
        f, cx, cy = intrinsics(W, H)
        f_full, cx_full, cy_full = intrinsics(int(round(W / ds.scale)), int(round(H / ds.scale)))
        ext = RT_BC
        sky = SKY
        n_cam = ds.n_cam
        stamp_of = lambda k: float(ds.cam_t[k])  # noqa: E731
        yy, xx = np.mgrid[0:int(round(H / ds.scale)), 0:int(round(W / ds.scale))]
        ray = np.sqrt(((xx - cx_full) / f_full) ** 2 + ((yy - cy_full) / f_full) ** 2 + 1.0)
        gt_range_working = lambda k: cv2.resize(ds.depth(k) * ray, (W, H), interpolation=cv2.INTER_NEAREST)  # noqa: E731
    cam = echo_li.PinholeCamera(f, f, cx, cy)
    pose = lambda k: T_nwu @ ds.pose(k) @ ext  # noqa: E731
    dz.SLANT_G[0] = np.tan(np.radians(args.slant_deg)) / f

    cfg = yaml.safe_load(open(args.config)) or {}
    sparse_cfg = cfg.get("SparseVog", {}) or {}
    settings = {k: sparse_cfg[k] for k in SPARSE_KEYS if k in sparse_cfg}
    settings["parametrization"] = "bearing_invdepth_additive3d"
    settings.setdefault("min_track_length", 1)
    settings["max_pool_size"] = args.max_pool
    for kv in args.sparse:
        k, v = kv.split("=", 1)
        settings[k] = type(sparse_cfg.get(k, 0.0))(v) if not isinstance(sparse_cfg.get(k), bool) else v.lower() == "true"
    fdf_settings = {}
    for kv in args.fdf:
        k, v = kv.split("=", 1)
        fdf_settings[k] = float(v)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg, tf)
        cfg_path = tf.name
    fcfg = echo_li.FrontendConfig.from_yaml(cfg_path)
    fcfg.set_camera(f, f, cx, cy, W, H, [])
    klt = echo_li.Frontend(fcfg, W, H)
    fdf = echo_li.FlowDepF(f, f, cx, cy, W, H, fdf_settings or None)
    sparse_klt = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)
    sparse_fdf = echo_li.Sparse3DFilter.bearing_invdepth_additive3d(cam, **settings)

    cm = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)[:, 0, ::-1]  # RGB rows, we use BGR below
    cm = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(256, 1), cv2.COLORMAP_JET)[:, 0, :]
    lo, hi = np.log(args.range_lo), np.log(args.range_hi)

    def range_col(r):
        t = (np.log(max(r, 1e-3)) - lo) / (hi - lo)
        c = cm[int(np.clip(1.0 - t, 0, 1) * 255)]  # jet reversed: near = red
        return (int(c[0]), int(c[1]), int(c[2]))

    def err_col(e):
        t = float(np.clip(e / 0.2, -1, 1))
        if t < 0:
            return (255, int(255 * (1 + t)), int(255 * (1 + t)))  # blue-ish (BGR) for short
        return (int(255 * (1 - t)), int(255 * (1 - t)), 255)  # red for long

    acc = {"klt": [], "fdf": []}
    dense_err = []
    dense_cov = []

    dense_nees = []
    def dense_panel_of(dense_r, covered, e_d, base, label):
        t_d = np.clip((np.log(np.clip(dense_r, 1e-3, None)) - lo) / (hi - lo), 0, 1)
        pnl = cm[((1.0 - t_d) * 255).astype(np.uint8)].copy()
        pnl[~covered] = (base[~covered] * 0.35).astype(np.uint8)
        pnl = np.ascontiguousarray(pnl)
        txt(pnl, f"{label}: {100*covered.mean():.0f}% cov, med|err| {100*np.median(np.abs(e_d)) if e_d.size else 0:.1f}%", (6, 16))
        return pnl
    raw_out = args.out.replace(".mp4", "_mp4v.mp4")
    vw = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (4 * W, 2 * H))
    n = min(args.frames, n_cam - args.start)
    for i in range(n):
        k = args.start + i
        gray = np.ascontiguousarray(ds.image(k))
        T_wc = pose(k)
        feats, _ = klt.process(gray)
        uv_klt = {int(fd["id"]): (float(fd["x"]), float(fd["y"])) for fd in feats}
        uv_fdf = fdf.process(gray, T_wc.tolist())
        stamp = stamp_of(k)
        sparse_klt.update(stamp, uv_klt, T_wc.tolist(), None, None)
        sparse_fdf.update(stamp, uv_fdf, T_wc.tolist(), None, None)
        # Posterior feedback for FlowDep-F (every feature, converged or not).
        post = {}
        for fid, fd in sparse_fdf.get_features().items():
            est = np.asarray(fd["position"], float)
            cov = np.asarray(fd["covariance_euclidean"], float)
            rr = float(np.linalg.norm(est))
            if rr > 1e-6 and np.isfinite(cov).all():
                rh = est / rr
                vr = float(rh @ cov @ rh)
                if vr > 0:
                    post[int(fid)] = (rr, float(np.sqrt(vr)))
        if post:
            fdf.set_posterior(post)

        gt_s = gt_range_working(k)
        base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        panels = {}
        stats = {}
        for name, sp, uvs in (("klt", sparse_klt, uv_klt), ("fdf", sparse_fdf, uv_fdf)):
            pr = base.copy()
            pe = base.copy()
            errs = []
            for fid, fd in sp.get_features().items():
                if sp.query_range(int(fid))[0] <= 0.0:
                    continue
                est = np.asarray(fd["position"], float)
                re = float(np.linalg.norm(est))
                if fid in uvs:
                    px, py = uvs[fid]
                elif est[2] > 0.1:
                    px, py = f * est[0] / est[2] + cx, f * est[1] / est[2] + cy
                else:
                    continue
                gx, gy = int(round(px)), int(round(py))
                if not (2 <= gx < W - 2 and 2 <= gy < H - 2):
                    continue
                gt = float(gt_s[gy, gx])
                p = (int(round(px)), int(round(py)))
                cv2.circle(pr, p, 2, range_col(re), -1, cv2.LINE_AA)
                if 0.5 < gt < sky:
                    e = (re - gt) / gt
                    errs.append(e)
                    if abs(e) > 0.2:
                        cv2.drawMarker(pe, p, err_col(e), cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA)
                    else:
                        cv2.circle(pe, p, 2, err_col(e), -1, cv2.LINE_AA)
            errs = np.array(errs)
            acc[name].append(errs)
            stats[name] = (len(errs), np.median(np.abs(errs)) * 100 if len(errs) else np.nan,
                           np.mean(np.abs(errs) > 0.2) * 100 if len(errs) else np.nan)
            panels[name] = (pr, pe)
        # GT panel.
        g = gt_s
        t = np.clip((np.log(np.clip(np.nan_to_num(g, nan=0.0), 1e-3, None)) - lo) / (hi - lo), 0, 1)
        idx = ((1.0 - t) * 255).astype(np.uint8)
        gt_panel = cm[idx]
        gt_panel[~np.isfinite(g) | (g <= 0.5) | (g >= sky)] = (40, 40, 40)
        gt_panel = np.ascontiguousarray(gt_panel)

        # Stats panel.
        sp_panel = np.zeros((H, W, 3), np.uint8)
        all_k = np.concatenate(acc["klt"]) if acc["klt"] else np.zeros(0)
        all_f = np.concatenate(acc["fdf"]) if acc["fdf"] else np.zeros(0)
        y = 22
        txt(sp_panel, f"{args.dataset} {args.traj} frame {k}  (GT pose, Sparse3D converged only)", (6, y)); y += 22
        txt(sp_panel, "this frame        n   med|err|  gross>20%", (6, y)); y += 18
        for name, lab in (("klt", "KLT      "), ("fdf", "FlowDep-F")):
            cnt, med, gr = stats[name]
            txt(sp_panel, f"{lab}   {cnt:4d}   {med:5.1f}%    {gr:4.1f}%", (6, y)); y += 18
        y += 8
        txt(sp_panel, "running (all frames so far)   n    med|err|  p90   gross", (6, y)); y += 18
        for lab, e in (("KLT      ", all_k), ("FlowDep-F", all_f)):
            if len(e):
                txt(sp_panel, f"{lab}  {len(e):6d}  {100*np.median(np.abs(e)):5.1f}%  {100*np.percentile(np.abs(e),90):5.1f}%  {100*np.mean(np.abs(e)>0.2):4.1f}%", (6, y))
            y += 18
        y += 8
        txt(sp_panel, f"colour: range {args.range_lo:.0f} m (red) .. {args.range_hi:.0f} m (blue), log", (6, y)); y += 18
        txt(sp_panel, "error: blue short, red long, x = |err| > 20%", (6, y)); y += 18
        txt(sp_panel, f"FlowDep-F live {fdf.n_live()}  observed {int(fdf.stats()['observed'])}  KLT tracked {len(uv_klt)}", (6, y)); y += 26
        de = np.concatenate(dense_err) if dense_err else np.zeros(0)
        dn = np.concatenate(dense_nees) if dense_nees else np.zeros(0)
        txt(sp_panel, f"densified ({args.dense_rule}, slant {args.slant_deg:.0f} deg), running", (6, y)); y += 18
        if de.size:
            txt(sp_panel, f"  coverage {100*np.nanmean(dense_cov):4.1f}%  med|err| {100*np.median(np.abs(de)):4.1f}%  p90 {100*np.percentile(np.abs(de),90):4.1f}%  gross {100*np.mean(np.abs(de)>0.2):4.1f}%", (6, y)); y += 18
            txt(sp_panel, f"  per-pixel NEES med {np.median(dn):.2f}  >chi2_95 {100*np.mean(dn>3.84):4.1f}%  >chi2_99 {100*np.mean(dn>6.63):4.1f}%", (6, y)); y += 18

        # Densified range frame from the converged Sparse3D patches, rule of row 5b.
        lev = {o[0]: o[3] for o in fdf.last_observations()}
        items = []
        for fid, fd in sparse_fdf.get_features().items():
            if sparse_fdf.query_range(int(fid))[0] <= 0.0 or fid not in uv_fdf:
                continue
            est = np.asarray(fd["position"], float)
            cov = np.asarray(fd["covariance_euclidean"], float)
            rr = float(np.linalg.norm(est))
            if rr <= 1e-6 or not np.isfinite(cov).all():
                continue
            rh = est / rr
            var_c = float(rh @ cov @ rh) / (rr * rr)
            var_rep = float(fd["range_var_reported"]) / (rr * rr)
            if var_c > 0:
                items.append((uv_fdf[fid][0], uv_fdf[fid][1], lev.get(fid, 0), np.log(rr), var_c, float(np.sqrt(max(var_rep - var_c, 0.0)))))
        if items:
            eta_p, var_p, covered = dz.fuse(items, H, W)[args.dense_rule]
        else:
            eta_p = np.zeros((H, W)); var_p = np.ones((H, W)); covered = np.zeros((H, W), bool)
        filled = np.zeros((H, W), bool)
        if args.fill and covered.any():
            ef, vf, filled, _ = dz.fill(eta_p, var_p, covered)
            eta_p = np.where(filled, ef, eta_p); var_p = np.where(filled, vf, var_p)
            covered = covered | filled
        dense_r = np.where(covered, np.exp(eta_p), 0.0)
        okgt = covered & np.isfinite(gt_s) & (gt_s > 0.5) & (gt_s < sky)
        e_d = (dense_r[okgt] - gt_s[okgt]) / gt_s[okgt]
        n_d = (eta_p[okgt] - np.log(gt_s[okgt])) ** 2 / np.maximum(var_p[okgt], 1e-12)
        dense_err.append(e_d); dense_cov.append(covered.mean()); dense_nees.append(n_d)
        dense_panel = dense_panel_of(dense_r, covered, e_d, base, "densified range")
        if filled.any():
            dense_panel[filled] = (0.55 * dense_panel[filled] + 0.45 * 255).astype(np.uint8)
            txt(dense_panel, f"measured {100*np.mean(covered & ~filled):.0f}% + filled {100*np.mean(filled):.0f}% (paler)", (6, 34))
        # Signed error panel of the dense frame.
        derr_panel = (base * 0.35).astype(np.uint8)
        e_full = np.zeros((H, W)); e_full[okgt] = (dense_r[okgt] - gt_s[okgt]) / gt_s[okgt]
        t_e = np.clip(e_full / 0.2, -1, 1)
        col = np.zeros((H, W, 3), np.uint8)
        neg = t_e < 0
        col[..., 0] = 255; col[..., 1] = (255 * (1 - np.abs(t_e))).astype(np.uint8); col[..., 2] = (255 * (1 - np.abs(t_e))).astype(np.uint8)
        col[~neg] = np.stack([(255 * (1 - t_e))[~neg], (255 * (1 - t_e))[~neg], np.full((~neg).sum(), 255)], axis=1).astype(np.uint8)
        derr_panel[okgt] = col[okgt]
        txt(derr_panel, f"densified signed error (this frame med|err| {100*np.median(np.abs(e_d)) if e_d.size else 0:.1f}%)", (6, 16))
        # Per-pixel sigma panel (log-range sigma in %, 1..30 % log scale).
        sig_panel = (base * 0.35).astype(np.uint8)
        s_rel = np.sqrt(np.maximum(var_p, 1e-12))
        t_s = np.clip((np.log(np.maximum(s_rel, 1e-4)) - np.log(0.01)) / (np.log(0.30) - np.log(0.01)), 0, 1)
        sig_col = cm[((1.0 - t_s) * 255).astype(np.uint8)]
        sig_panel[covered] = sig_col[covered]
        txt(sig_panel, f"densified sigma: red 1% .. blue 30%  (this frame med {100*np.median(s_rel[covered]) if covered.any() else 0:.1f}%)", (6, 16))
        klt_all = base.copy()
        for (px, py) in uv_klt.values():
            cv2.circle(klt_all, (int(round(px)), int(round(py))), 2, (200, 200, 200), 1, cv2.LINE_AA)
        txt(klt_all, f"KLT tracked: {len(uv_klt)}", (6, 16))
        for name, lab in (("klt", "KLT converged range"), ("fdf", "FlowDep-F converged range")):
            txt(panels[name][0], lab, (6, 16))
            txt(panels[name][1], lab.replace("converged range", "signed error"), (6, 16))
        txt(gt_panel, "GT range", (6, 16))
        # Stats text goes on the sigma panel's lower half.
        sp_small = sp_panel[: H // 2]
        sig_panel[H // 2:] = cv2.addWeighted(sig_panel[H // 2:], 0.25, sp_small, 1.0, 0)
        top = np.hstack([panels["klt"][0], panels["fdf"][0], dense_panel, gt_panel])
        bot = np.hstack([panels["klt"][1], panels["fdf"][1], derr_panel, sig_panel])
        frame = np.vstack([top, bot])
        vw.write(frame)
        if args.png_every and i % args.png_every == 0:
            cv2.imwrite(args.out.replace(".mp4", f"_f{k}.png"), frame)
        if i % 20 == 0:
            print(f"  [{i}/{n}] klt {stats['klt'][0]}  fdf {stats['fdf'][0]}", flush=True)
    vw.release()
    # Browser-playable H.264 (mp4v from OpenCV is not).
    import subprocess
    rc = subprocess.call(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "20", "-movflags", "+faststart", args.out])
    if rc == 0:
        import os
        os.remove(raw_out)
    else:
        print(f"ffmpeg failed ({rc}); mp4v file kept at {raw_out}")
    all_k = np.concatenate(acc["klt"])
    all_f = np.concatenate(acc["fdf"])
    for lab, e in (("KLT", all_k), ("FlowDep-F", all_f)):
        print(f"{lab}: n {len(e)}  per frame {len(e)/n:.1f}  med|err| {100*np.median(np.abs(e)):.1f}%  p90 {100*np.percentile(np.abs(e),90):.1f}%  gross {100*np.mean(np.abs(e)>0.2):.1f}%")
    de = np.concatenate(dense_err); dn = np.concatenate(dense_nees)
    print(f"densified frame ({args.dense_rule}, slant {args.slant_deg:.0f}): coverage mean {100*np.nanmean(dense_cov):.1f}% (p10 {100*np.nanpercentile(dense_cov,10):.1f}%, p90 {100*np.nanpercentile(dense_cov,90):.1f}%)  per-pixel med|err| {100*np.median(np.abs(de)):.1f}%  p90 {100*np.percentile(np.abs(de),90):.1f}%  gross {100*np.mean(np.abs(de)>0.2):.1f}%  NEES med {np.median(dn):.2f}  >3.84 {100*np.mean(dn>3.84):.1f}%  >6.63 {100*np.mean(dn>6.63):.1f}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
