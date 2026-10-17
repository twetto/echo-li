"""Extract one Pi4-rig rosbag2 (sqlite) into sensors.npz + frames320.bin.

Same conventions as the bikevid extraction (bike_sensors_hdr.npz): image stamps are the
HEADER stamp (= hardware trigger, bikevid-stamps-header-is-trigger), not bag receive
time; IMU stamps are header stamps; odometry = the rig's own /echo_li/odometry.
Frames are decoded with IMREAD_REDUCED_GRAYSCALE_4 like the live rig and mkcache.py.
Runs inside the ubuntu-22-04 distrobox (needs rclpy for CDR decoding).
"""
import sys, os, sqlite3, glob, numpy as np, cv2
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, CameraInfo, CompressedImage
from nav_msgs.msg import Odometry

bagdir, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
# A long recording is split into <bag>_0.db3, _1.db3, ... (pickup_20261001: three files);
# each file carries its own topic table, so topic ids are resolved per file.
dbs = [(sqlite3.connect(f"file:{db}?mode=ro", uri=True))
       for db in sorted(glob.glob(os.path.join(bagdir, "*.db3")))]
tids = [{n: i for i, n, _ in c.execute("select id,name,type from topics")} for c in dbs]
tid = {n for t in tids for n in t}
W, H = 320, 200

def rows(topic):
    for c, t in zip(dbs, tids):
        if topic in t:
            yield from c.execute("select timestamp, data from messages where topic_id=? order by timestamp", (t[topic],))

imu = []
for _, d in rows("/imu"):
    m = deserialize_message(bytes(d), Imu)
    s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    imu.append((s, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z,
                m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
imu = np.array(imu)
if len(imu) == 0:
    sys.exit(f"{bagdir}: no /imu messages in {len(dbs)} db3 file(s); topics {sorted(tid)}")

odom = []
if "/echo_li/odometry" in tid:
    for _, d in rows("/echo_li/odometry"):
        m = deserialize_message(bytes(d), Odometry)
        s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        p, q = m.pose.pose.position, m.pose.pose.orientation
        odom.append((s, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
odom = np.array(odom) if odom else np.zeros((0, 8))

K = Dc = None; cw = ch = 0; model = ""
# The topic may be declared with no messages at all (pickup_20261001 bag), or carry K = 0.
row = next(rows("/camera/camera_info"), None)
if row is not None:
    ci = deserialize_message(bytes(row[1]), CameraInfo)
    K = np.array(ci.k, float).reshape(3, 3); Dc = np.array(ci.d, float); cw, ch = ci.width, ci.height; model = ci.distortion_model

img_t, img_t_recv, good, bad = [], [], [], []
src_shape = None
with open(os.path.join(outdir, "frames320.bin"), "wb") as f:
    for n, (trecv, d) in enumerate(rows("/camera/image/compressed")):
        m = deserialize_message(bytes(d), CompressedImage)
        s = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        buf = np.frombuffer(bytes(m.data), np.uint8)
        if src_shape is None:
            full = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if full is None:
                bad.append(n); continue
            src_shape = full.shape
            # 1280x800 source: the rig's own /4 decode. 640x400 source (10-16 bags): /2, so
            # the cache is a true 320x200 decode and not a 160x100 decode blown back up.
            reduce = {1280: cv2.IMREAD_REDUCED_GRAYSCALE_4, 640: cv2.IMREAD_REDUCED_GRAYSCALE_2}.get(
                src_shape[1], cv2.IMREAD_GRAYSCALE)
            print(f"  source {src_shape[1]}x{src_shape[0]} -> decode flag {reduce}", flush=True)
        g = cv2.imdecode(buf, reduce)
        if g is None:
            bad.append(n); continue
        if g.shape[0] != H or g.shape[1] != W:
            g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)
        f.write(np.ascontiguousarray(g[:H, :W]).tobytes())
        good.append(n); img_t.append(s); img_t_recv.append(trecv * 1e-9)
        if n and n % 5000 == 0:
            print("  ", n, flush=True)
img_t = np.array(img_t); img_t_recv = np.array(img_t_recv)
if len(img_t) == 0:
    sys.exit(f"{bagdir}: no decodable /camera/image/compressed frames")
np.save(os.path.join(outdir, "frames320_index.npy"), np.array(good))
np.savez(os.path.join(outdir, "sensors.npz"), imu=imu, img_t=img_t, img_t_recv=img_t_recv, odom=odom,
         K=K if K is not None else np.zeros((3, 3)), dist=Dc if Dc is not None else np.zeros(0),
         width=cw, height=ch, distortion_model=model, src_shape=np.array(src_shape or (0, 0)))
print(f"{os.path.basename(bagdir)}: imu {imu.shape} img {len(img_t)} (bad {len(bad)}) odom {odom.shape} src {src_shape} "
      f"camera_info {cw}x{ch} {model}")
if K is not None:
    print(f"  K fx {K[0,0]:.3f} fy {K[1,1]:.3f} cx {K[0,2]:.3f} cy {K[1,2]:.3f}  D {np.round(Dc,6).tolist()}")
print(f"  imu dt median {np.median(np.diff(imu[:,0]))*1e3:.2f} ms  |a| median {np.median(np.linalg.norm(imu[:,4:7],axis=1)):.3f}"
      f"  img dt median {np.median(np.diff(img_t))*1e3:.2f} ms  recv-header median {np.median(img_t_recv-img_t)*1e3:.1f} ms"
      f"  first img - first imu {img_t[0]-imu[0,0]:.2f} s  odom span {odom[0,0]-img_t[0] if len(odom) else float('nan'):.2f}..{odom[-1,0]-img_t[-1] if len(odom) else float('nan'):.2f} s")
