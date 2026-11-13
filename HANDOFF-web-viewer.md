# Handoff: remote web viewer for the ECHO-LI rig

Goal: watch a live ECHO-LI run (tracker overlay, trajectory, landmarks) in a browser on any machine,
while the VIO runs on the rig. Target for this weekend: the Pi4 rig that already runs
`voxl2_vio_node` and publishes ROS 2 topics. Android as a rig is a later project (notes at the end).

Nothing below has been run end to end yet. Items marked **verified** were checked in a cloud
session; everything else is a plan and should be checked as it is built.

## Decisions made

- **The VIO stays on the rig.** The browser only displays. No wasm port of the VIO is needed.
- **Viewer: Rerun's web viewer**, same as the CLI's `--vis`. The repo pins Rerun **0.31.3**, and
  the rig side, the server and the viewer must all use that exact version.
- **ROS traffic stays on the Pi.** A small bridge node on the Pi turns ROS topics into a Rerun
  stream. DDS never crosses the internet.
- **The Linode is the relay:** 1 GB RAM, nginx already installed, domain already pointing at it.
- **The Pi reaches the Linode over Tailscale.** Browsers reach the Linode over HTTPS with a password.

```
Pi4: voxl2_vio_node ──ROS (local)──► rerun_bridge.py ──gRPC over Tailscale──► Linode :9876
                                                                                    │
browsers ──HTTPS + basic auth──► nginx (vio.<domain>) ──► rerun --serve-web (:9090 viewer, :9876 gRPC)
```

## Work items

### 1. Linode (about an hour)
- [ ] DNS: `A` record `vio.<domain>` pointing at the Linode, unless a wildcard record already covers it.
- [ ] `sudo certbot --nginx -d vio.<domain>`
- [ ] 1 GB swap file, as a safety net.
- [ ] `pip install rerun-sdk==0.31.3` (provides the `rerun` binary). **Do not compile anything on
      the Linode:** Rust builds will run out of memory on 1 GB.
- [ ] A systemd unit running something like
      `rerun --serve-web --port 9876 --web-viewer-port 9090 --server-memory-limit 300MB --bind <addr>`.
      Check the exact flag names with `rerun --help` for 0.31.3.
      - The Pi pushes straight to `rerun+http://<linode-tailscale-ip>:9876/proxy`, bypassing nginx.
        So 9876 has to listen on the Tailscale interface. Firewall it so it is reachable only on
        `tailscale0` and from localhost, never from the public internet.
      - 9090 (the viewer page) only needs to be reachable from localhost (nginx).
- [ ] An `htpasswd` file and an nginx site config. Sketch:

```nginx
server {
    listen 443 ssl http2;
    server_name vio.<domain>;
    # ssl_* lines written by certbot

    auth_basic           "echo-li";
    auth_basic_user_file /etc/nginx/echo-li.htpasswd;

    location /rerun. {                    # Rerun gRPC calls from the in-browser viewer
        proxy_pass         http://127.0.0.1:9876;
        proxy_http_version 1.1;
        proxy_buffering    off;           # the stream never ends
        proxy_read_timeout 1h;
    }
    location / {                          # the viewer page itself
        proxy_pass http://127.0.0.1:9090;
    }
}
```

  Viewer link: `https://vio.<domain>/?url=rerun%2Bhttps://vio.<domain>/proxy`

  **Unverified, test first:**
  - Do all of the gRPC paths the viewer calls really start with `/rerun.`? Check with the browser's
    developer tools or the nginx access log, and adjust the `location` if not.
  - Does the viewer accept a `rerun+https://…/proxy` URL behind a proxy?
  - Does the browser reuse the basic-auth password for the viewer's fetch calls on the same origin?

  Do this first on Saturday. A quick check: stream a EuRoC run from any machine with the existing
  CLI, `ECHO_LI_RERUN_URL=rerun+http://<linode>:9876/proxy cargo run -p echo-li-cli --release --features rerun -- -d … -c configs/eqvio_euroc_euclid.yaml --vis`
  (see `echo-li-cli/src/main.rs:1197`), then open the viewer link.

**RAM budget:** OS + nginx ~150 MB, the rerun process ~100 MB (an estimate; measure it), plus the
buffer set by `--server-memory-limit` (300 MB). When the buffer fills, the oldest data is dropped,
so viewers who join late still see recent history.

### 2. Tailscale (about 10 minutes)
- [ ] Install Tailscale on the Pi and on the Linode, and note the Linode's `100.x` IP.

### 3. `rerun_bridge.py` on the Pi (about half a day)
Python, `rclpy` + `rerun-sdk==0.31.3`, connecting with `rr.connect_grpc("rerun+http://<linode-ts-ip>:9876/proxy")`.

- [ ] **Check first:** `pip install rerun-sdk==0.31.3` works on the Pi (Humble means Ubuntu 22.04
      arm64, Python 3.10). If it doesn't, run the bridge on the Linode instead and carry ROS over
      Tailscale with zenoh or `foxglove_bridge`. That is less nice.

Topics, from `echo-li-ros2/src/bin/voxl2_vio_node.rs` (default parameters around line 190;
publishers around line 1905):

| ROS topic | Message | Rerun |
|---|---|---|
| `/echo_li/odometry` | `nav_msgs/Odometry` | `Transform3D` on the camera/body entity |
| `/echo_li/path` | `nav_msgs/Path` | `LineStrips3D` |
| `/echo_li/landmarks` | `PointCloud2` | `Points3D` |
| `/echo_li/sparse3d_landmarks` | `PointCloud2` | `Points3D` |
| `/echo_li/occupancy` (if enabled) | `PointCloud2` | `Boxes3D` or `Points3D` |
| `/tf` (if `publish_tf`) | `tf2_msgs/TFMessage` | optional |
| camera topic (`image_topic` parameter; default `/tracking_front/decoded`, Pi rig may differ; `image_transport` can be `raw` or `compressed`) | `Image` / `CompressedImage` | `EncodedImage` (JPEG), throttled to 2–5 Hz |
| **new** tracked-points topic (item 4) | see item 4 | `Points2D` on the image entity |

- [ ] Log with a time axis taken from the ROS header stamps (`rr.set_time(...)`).
- [ ] Send a blueprint that mirrors the CLI layout (`send_rerun_blueprint` in
      `echo-li-cli/src/main.rs:491`): a 3D view, plus a 2D image view with the tracked points
      drawn over the image.
- [ ] Mind the Pi's CPU: JPEG-encode only at the throttled rate. If the camera topic is already
      compressed, pass the JPEG bytes through without re-encoding. Watch the VIO frame timing while
      the bridge runs.
- [ ] Run the bridge as a systemd service or from the existing launch script.

### 4. Tracked-points topic in `voxl2_vio_node` (a few hours)
The node does not publish the tracker state yet. The data is right there:
`voxl2_vio_node.rs:1001-1003`, in `process_image`: `self.frontend.process(&rudolf_img)` → `features: Vec<Feature>`
(`f.id`, `f.x`, `f.y`).

- [ ] Publish them at a few Hz, for example as a `PointCloud2` with fields `x`, `y`, `id` (z = 0) on
      `/echo_li/tracks`, carrying the **same header stamp as the image** so the bridge can match
      the two.
- [ ] **Watch `image_scale`** (parameter, 0.25–1.0): feature coordinates are in the scaled image.
      Either rescale them to full resolution, or have the bridge log the image at the same scale.
- [ ] For reference, the CLI draws the same points in `echo-li-cli/src/main.rs:1518` (with
      `clip_image_point`) and logs them at 1789.
- [ ] Optional: colour each point by track age or by inlier/outlier status, if that is easy to get.

### 5. Polish (Sunday)
- [ ] Start everything on boot, tune `--server-memory-limit`, pick the image rate for the Pi's upload speed.

**Zero-code fallback if the Rerun route stalls:** `foxglove_bridge` on the Pi plus Lichtblick (the
open-source Foxglove Studio fork) hosted behind nginx. That shows the trajectory and landmarks with
no code; the tracker overlay still needs item 4.

## Later: Android as the rig (not this weekend)
Findings from a cloud session; only compile checks, nothing linked or run.

- **verified:** `echo-li-core` compiles for `aarch64-linux-android` once Rudolf-V's `minifb`
  dependency is made optional. `minifb` is the live debug window. It is unconditional in Rudolf-V's
  `Cargo.toml` but only its examples use it, and its Wayland/X11 dependencies break the Android and
  wasm builds. Fix it upstream in Rudolf-V (`minifb` optional, behind a `viz` feature).
- **verified:** `echo-li-core` also compiles for `wasm32-unknown-unknown` with:
  - `getrandom` 0.3 with feature `wasm_js` and `--cfg getrandom_backend="wasm_js"`;
  - `--cfg=web_sys_unstable_apis`, which wgpu needs;
  - a newer `js-sys`/`wasm-bindgen` than the lockfile has. `rerun` 0.31 holds them back, so a wasm
    build needs its own workspace.

  At run time in a browser, `Instant::now()` panics (Rudolf-V `frontend.rs`, `gpu/*`; use the
  `web-time` crate) and `pollster::block_on` in `gpu/device.rs` cannot block the browser's main thread.
- **partly checked:** the Rerun SDK with the `server` and `web_viewer` features was checked for
  Android. All the Rust code got through; the build stopped at the `ring` crate, which needs the
  Android NDK's clang. The NDK couldn't be downloaded in the sandbox. Confirm in CI: GitHub Actions
  runners ship the NDK.
- Plan when it's time:
  1. A streaming API in `echo-li-core`: config from text, `push_imu`/`push_image`, no file I/O.
  2. A shared `echo-li-viz` crate holding the Rerun logging, pulled out of `echo-li-cli/src/main.rs`.
  3. An `echo-li-android` JNI `cdylib` built with `cargo-ndk`, pushing to the same Linode relay.
  4. A Kotlin app: CameraX Y plane, `SensorManager` at `SENSOR_DELAY_FASTEST`, and a check that
     `SENSOR_INFO_TIMESTAMP_SOURCE == REALTIME`.
  5. Calibrate with Kalibr per phone model; lock focus and turn off optical image stabilisation (OIS).

## Open questions for the user
- The exact subdomain.
- The Pi rig's camera topic name and `image_transport` setting. Probably in the launch script or
  `echo-li-ros2/config/eqvio_myrig.yaml`, but not confirmed.
