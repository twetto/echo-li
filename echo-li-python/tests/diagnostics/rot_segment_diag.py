"""Find the large-rotation window(s) in EuRoC V1_03_difficult from GT attitude.

Angular rate is computed two ways and cross-checked:
  (a) GT quaternion finite-difference  -> smooth true body rate
  (b) IMU gyro minus GT gyro-bias      -> sanity check
Reports the top rotation segments and the near-end window the user recalls.
"""
import csv, os
import numpy as np

HOME = os.path.expanduser("~")
D = os.path.join(HOME, "Downloads", "vicon_room1", "vicon_room1",
                 "V1_03_difficult", "mav0")

def load_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    return np.array(rows, dtype=float)

gt = load_csv(os.path.join(D, "state_groundtruth_estimate0", "data.csv"))
t = gt[:, 0] * 1e-9          # s
t0 = t[0]
tr = t - t0
quat = gt[:, 4:8]            # w, x, y, z
bw = gt[:, 11:14]            # gyro bias

# --- (a) angular rate from GT quaternion finite difference ---
def qmul(a, b):
    aw, ax, ay, az = a.T
    bw_, bx, by, bz = b.T
    return np.stack([
        aw*bw_ - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw_ + ay*bz - az*by,
        aw*by - ax*bz + ay*bw_ + az*bx,
        aw*bz + ax*by - ay*bx + az*bw_], axis=1)

qc = quat.copy(); qc[:, 1:] *= -1.0            # conjugate
dq = qmul(qc[:-1], quat[1:])                    # relative rotation q_k^-1 q_{k+1}
dq /= np.linalg.norm(dq, axis=1, keepdims=True)
ang = 2.0 * np.arccos(np.clip(np.abs(dq[:, 0]), -1, 1))   # rotation angle per step
dt = np.diff(t)
w_gt = ang / dt                                 # rad/s magnitude
w_gt = np.concatenate([w_gt, w_gt[-1:]])

# --- (b) IMU gyro minus bias (interpolate bias onto imu stamps) ---
imu = load_csv(os.path.join(D, "imu0", "data.csv"))
ti = imu[:, 0] * 1e-9
wimu = imu[:, 1:4]
bwi = np.stack([np.interp(ti, t, bw[:, j]) for j in range(3)], axis=1)
w_imu = np.linalg.norm(wimu - bwi, axis=1)

dur = tr[-1]
print(f"V1_03_difficult: duration {dur:.1f}s, {len(t)} GT samples @ "
      f"{1/np.median(dt):.0f} Hz")
print(f"GT body angular rate  |w|: median {np.median(w_gt):.2f}, "
      f"p90 {np.percentile(w_gt,90):.2f}, max {w_gt.max():.2f} rad/s")
print(f"IMU gyro (bias-corr)  |w|: median {np.median(w_imu):.2f}, "
      f"p90 {np.percentile(w_imu,90):.2f}, max {w_imu.max():.2f} rad/s "
      f"({np.degrees(w_imu.max()):.0f} deg/s)")

# --- smooth GT rate (0.5 s box) and find high-rotation segments ---
win = max(1, int(0.5 / np.median(dt)))
sm = np.convolve(w_gt, np.ones(win)/win, mode="same")
thr = np.percentile(sm, 90)
hot = sm > thr
# contiguous runs
segs = []
i = 0
while i < len(hot):
    if hot[i]:
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        segs.append((i, j))
        i = j
    else:
        i += 1
segs = [(a, b) for a, b in segs if tr[b-1]-tr[a] > 0.5]   # >0.5 s
segs.sort(key=lambda s: -sm[s[0]:s[1]].max())

print(f"\nhot segments (|w| smoothed > p90={thr:.2f} rad/s), by peak rate:")
print(f"  {'t_start':>8} {'t_end':>8} {'dur':>6} {'peak':>6} {'mean':>6}  abs_ts_start..end")
for a, b in segs[:8]:
    peak = sm[a:b].max(); mean = sm[a:b].mean()
    print(f"  {tr[a]:8.1f} {tr[b-1]:8.1f} {tr[b-1]-tr[a]:6.1f} "
          f"{peak:6.2f} {mean:6.2f}  {int(t[a]*1e9)}..{int(t[b-1]*1e9)}")

# --- near-end focus (last 25%) ---
end_mask = tr > 0.75 * dur
ei = np.argmax(sm * end_mask)
print(f"\nnear-end peak rotation: t={tr[ei]:.1f}s  |w|={sm[ei]:.2f} rad/s "
      f"({np.degrees(sm[ei]):.0f} deg/s), abs_ts={int(t[ei]*1e9)}")

# --- rotation-DOMINANCE: high w, low translation (parallax starvation) ---
vel = gt[:, 8:11]
speed = np.linalg.norm(vel, axis=1)                     # m/s
sm_v = np.convolve(speed, np.ones(win)/win, mode="same")
# parallax-to-rotation ratio: (|v|/Z)/w in px-flow terms; use Z~2.5 m (Vicon room)
Z = 2.5
par_ratio = (sm_v / Z) / np.maximum(sm, 1e-3)           # <<1 => rotation-dominated
print(f"\ntranslation speed |v|: median {np.median(speed):.2f}, "
      f"p10 {np.percentile(speed,10):.2f} m/s")
print(f"parallax/rotation ratio (|v|/Z)/w, Z={Z}m: "
      f"median {np.median(par_ratio):.2f}, p10 {np.percentile(par_ratio,10):.2f}")

# rotation-dominant = w in top 25% AND ratio in bottom 25%
rd = (sm > np.percentile(sm, 75)) & (par_ratio < np.percentile(par_ratio, 25))
segs2 = []
i = 0
while i < len(rd):
    if rd[i]:
        j = i
        while j < len(rd) and rd[j]:
            j += 1
        if tr[j-1]-tr[i] > 0.3:
            segs2.append((i, j))
        i = j
    else:
        i += 1
segs2.sort(key=lambda s: -sm[s[0]:s[1]].max())
print(f"\nrotation-DOMINANT segments (high w & low parallax), by peak w:")
print(f"  {'t_start':>8} {'t_end':>8} {'dur':>6} {'w_pk':>6} {'v_mean':>7} {'ratio':>6}")
for a, b in segs2[:8]:
    print(f"  {tr[a]:8.1f} {tr[b-1]:8.1f} {tr[b-1]-tr[a]:6.1f} {sm[a:b].max():6.2f} "
          f"{sm_v[a:b].mean():7.2f} {par_ratio[a:b].mean():6.2f}")
