# ECHO-LI

Rust rewrite / experiment branch for ECHO-LI, including EqVIO, Rudolf-V tracking integration, sparse out-of-state depth filtering, and patch depth mapping.

## Common Commands

Run the default EuRoC visual demo. This is what `run.sh` currently contains:

```bash
cargo run -p echo-li-cli --release --features rerun -- \
  -d /path/to/V1_01_easy \
  -c configs/eqvio_euroc_euclid.yaml \
  --vis
```

Profile the CLI with Samply. This is what `profile.sh` currently contains:

```bash
cargo build -p echo-li-cli --release --features rerun
samply record ./target/release/echo-li-cli \
  -d /path/to/V1_01_easy \
  --config configs/eqvio_euroc_euclid.yaml \
  --vis
```

Enable multi-thread mode by `--features (rerun,)parallel`.

## Checks

```bash
cargo check -p echo-li-cli
cargo check -p echo-li-cli --features parallel
cargo check -p echo-li-cli --features rerun
cargo test -p echo-li-core
```

## Features

- `rerun`: enables Rerun visualization and `--vis`.
- `parallel`: enables Rayon paths in ECHO-LI and Rudolf-V's parallel feature.

## Notes

The default config used by the helper scripts is:

```text
configs/eqvio_euroc_euclid.yaml
```

The release profile keeps debug symbols enabled so Samply profiles can be symbolicated.
