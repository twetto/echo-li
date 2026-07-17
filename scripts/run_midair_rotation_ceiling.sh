#!/usr/bin/env bash
set -euo pipefail

# Plan B "comprehensive-motion ceiling": does any no-pose reference-anchored
# tracker beat the co-moving previous-frame template under 6-DOF (rotation)
# motion, and does an oracle pose+depth warp close the gap?
#
# This re-runs the reference-vs-previous-frame front-end ceiling on a
# ROTATION-HEAVY MidAir trajectory (as opposed to the near-forward-motion
# trajectory used by flow-bias result 15). See the write-up:
#   ../ECHO-LI-notes/docs/frontend/flow-bias/comprehensive_motion_ceiling.md
#
# The rotation-heavy VO_test slice is trajectory 0002 (sunny) / 2002 (sunset) /
# 1002 (foggy) -- the SAME path across atmospheres, ~2.6x the per-frame rotation
# of trajectory_0000 (p90 1.34 deg/frame vs 0.51), EuRoC-class peaks (~122 deg/s),
# rotation-dense throughout. This script prints the motion profile of the chosen
# slice first so you can confirm it is actually rotation-heavy before trusting
# the ceiling numbers.
#
# The Sparse3D depth-NEES and temporal/texture measurement cores for the same
# slice are reproduced by the existing suite, NOT here:
#   SLICES="sunny:2 sunset:2002 foggy:1002" bash scripts/run_midair_track_bias_suite.sh
#   PY=... midair_flow_texture_likelihood.py --cond sunny --traj 2 --config \
#       configs/diagnostics_midair_sparse3d.yaml   # texture core
#
# Run from the echo-li repo root (pure-Python KLT, ~10 min for the 600-frame trio):
#   bash scripts/run_midair_rotation_ceiling.sh
# Useful overrides:
#   ROOT=/path/to/MidAir COND=sunset TRAJ=2002 FRAMES=600 \
#       bash scripts/run_midair_rotation_ceiling.sh
#
# Headline (sunny traj_2): previous-frame beats every no-pose reference anchor by
# a WIDER margin than on forward motion; the oracle pose+depth planar warp only
# halves the coarse-reference displacement (6.4->3.0px vs fresh 0.71). No-pose
# reference-anchor search stays closed under 6-DOF motion.

PY="${PY:-echo-li-python/venv/bin/python}"
ROOT="${ROOT:-/home/twetto/Server250/18TB/datasets/dataset_MidAir/MidAir}"
SUBSET="${SUBSET:-VO_test}"
COND="${COND:-sunny}"
TRAJ="${TRAJ:-2}"                 # sunny:2  sunset:2002  foggy:1002 (same path)
FRAMES="${FRAMES:-600}"
SCALE="${SCALE:-0.5}"
N_VICTIMS="${N_VICTIMS:-16}"
OUTDIR="${OUTDIR:-/tmp/midair_rotation_ceiling_$(date +%Y%m%d_%H%M%S)}"

D="echo-li-python/tests/diagnostics"
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.log"
: > "$SUMMARY"

echo "Writing logs to $OUTDIR"
echo "slice: $SUBSET/$COND/traj_$(printf '%04d' "$TRAJ")  frames=$FRAMES scale=$SCALE"

# --- motion profile: confirm the chosen slice is rotation-heavy -----------------
echo
echo "===== motion profile (inter-camera-frame rotation, 25 Hz) ====="
"$PY" - "$ROOT" "$SUBSET" "$COND" "$TRAJ" <<'PYEOF' | tee -a "$SUMMARY"
import os, sys, numpy as np, h5py
from scipy.spatial.transform import Rotation as Rot
root, subset, cond, traj = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
h5 = os.path.join(root, subset, cond, "sensor_records.hdf5")
with h5py.File(h5, "r") as db:
    att = db[f"trajectory_{traj:04d}"]["groundtruth"]["attitude"][:]
qc = att[::4]  # 100 Hz GT -> 25 Hz camera
Rm = Rot.from_quat(np.column_stack([qc[:,1],qc[:,2],qc[:,3],qc[:,0]])).as_matrix()
rel = np.einsum('nij,nkj->nik', Rm[1:], Rm[:-1])
ang = np.degrees(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
print(f"  rot/frame deg  median={np.median(ang):.2f}  p90={np.percentile(ang,90):.2f}  "
      f"max={ang.max():.2f}  frames>1deg={100*np.mean(ang>1):.1f}%  "
      f"(peak {ang.max()*25:.0f} deg/s)")
print("  reference: trajectory_0000 forward-motion baseline is p90 0.51, 0% >1deg/frame")
PYEOF

run_case() {
  local name="$1"; shift
  local log="$OUTDIR/$name.log"
  echo
  echo "===== $name ====="
  "$@" > "$log" 2>&1 || echo "  ($name exited non-zero, see $log)"
  cat "$log"
}

# --- 1) prev-seeded reference: is it an init or an objective problem? -----------
run_case "prevseed" \
  "$PY" "$D/first_obs_prevseed.py" --root "$ROOT" --set "$SUBSET" --cond "$COND" \
  --traj "$TRAJ" --frames "$FRAMES" --scale "$SCALE" --out "$OUTDIR/prevseed"

# --- 2) coarse-fresh / fine-fixed hybrid lambda-sweep: best no-pose reference ---
run_case "hybrid" \
  "$PY" "$D/first_obs_hybrid.py" --root "$ROOT" --set "$SUBSET" --cond "$COND" \
  --traj "$TRAJ" --frames "$FRAMES" --scale "$SCALE" --out "$OUTDIR/hybrid"

# --- 3) oracle pose+depth homography warp: the ROVIO-class ceiling --------------
run_case "warptest" \
  "$PY" "$D/first_obs_warptest.py" --root "$ROOT" --set "$SUBSET" --cond "$COND" \
  --traj "$TRAJ" --frames "$FRAMES" --scale "$SCALE" --n-victims "$N_VICTIMS"

# --- summary --------------------------------------------------------------------
{
  echo; echo "===== prevseed (median |beta| by age; prev vs first vs pseed) ====="
  sed -n '/prev-seeded reference test/,/p90 |beta|/p' "$OUTDIR/prevseed.log" 2>/dev/null || true
  echo; echo "===== hybrid (delta vs previous-frame; negative beats prev) ====="
  sed -n '/delta vs previous-frame/,$p' "$OUTDIR/hybrid.log" 2>/dev/null || true
  echo; echo "===== warptest (VALID coarse L2: unwarp vs oracle-warp vs fresh) ====="
  grep -n "self-check\|MEDIAN \[VALID" "$OUTDIR/warptest.log" 2>/dev/null || true
} | tee -a "$SUMMARY"

echo
echo "Logs + npz written to $OUTDIR"
echo "Summary: $SUMMARY"
