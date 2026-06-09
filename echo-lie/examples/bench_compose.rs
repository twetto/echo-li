// Microbench: tight SO3::compose dependency chain to isolate the per-compose
// cost (A/B the renormalisation in compose). Sequential chain => can't be
// vectorised/elided; checksum printed to defeat dead-code elimination.
use echo_lie::SO3;
use nalgebra::Vector3;

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300_000_000);
    let step = SO3::exp(&Vector3::new(0.0011, 0.0023, -0.0017));
    let mut acc = SO3::identity();
    let mut sum = 0.0f64;
    for i in 0..n {
        acc = acc.compose(&step);
        if i & 0xFFFFF == 0 {
            sum += acc.as_xyzw()[0];
        }
    }
    println!("n={n} checksum={sum} q_norm={}", acc.q.norm());
}
