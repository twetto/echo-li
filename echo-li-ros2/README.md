# echo-li-ros2

ROS 2 **Humble** node. Humble is a 22.04 distro, so on a 24.04 (Noble) host it cannot be
installed natively — the native distro there is Jazzy, which these scripts do not support
(`run_voxl2_bag_humble.sh` and `run_nfu_bag.sh` both require `/opt/ros/humble/setup.bash`).
Build and run inside the `ubuntu-22-04` Distrobox instead.

## Creating the box

```sh
# amd64 explicitly: a cached ros:humble-ros-base may be arm64 from the Pi4 cross-build,
# and podman will happily run it under emulation without saying much.
podman pull --arch amd64 docker.io/library/ros:humble-ros-base

distrobox create --name ubuntu-22-04 \
  --image docker.io/library/ros:humble-ros-base \
  --additional-flags "--volume /mnt/18TB:/mnt/18TB"

distrobox enter ubuntu-22-04          # first enter runs distrobox-init; takes a few minutes
sudo apt-get update && sudo apt-get install -y libclang-dev clang
```

Two things that are easy to lose an hour to:

- **`--volume /mnt/18TB:/mnt/18TB` is required.** Distrobox shares `$HOME`, but `~/18TB` is
  a symlink to `/mnt/18TB/chen_fu_yeh`, so the repo resolves OUTSIDE `$HOME` and the symlink
  dangles inside the box without this mount.
- **`libclang-dev` is not in `ros:humble-ros-base`.** `r2r` generates its bindings with
  bindgen, which fails with `Unable to find libclang` until it is installed.

Rust comes from the host via the shared `$HOME` (`~/.cargo`, `~/.rustup`) — do not install a
second toolchain inside the box.

## Building

```sh
distrobox enter ubuntu-22-04
. /opt/ros/humble/setup.bash
CARGO_TARGET_DIR=target-humble cargo build --release -p echo-li-ros2 --features parallel
```

`CARGO_TARGET_DIR=target-humble` keeps the container's artifacts out of the host `target/`;
the bag scripts look for the binaries there.

Note the box is built from the ROS image rather than bare `ubuntu:22.04`, so Humble is
already present and no ROS apt install is needed. The name still matches what the bag
scripts tell you to enter; they only test for `setup.bash`.
