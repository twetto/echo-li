use std::time::Instant;

use echo_li_ros2::{NSEC_PER_SEC, best_effort_qos, stamp_ns};
use futures::StreamExt;
use r2r::sensor_msgs::msg::{Image, Imu};

#[derive(Clone, Copy, PartialEq, Eq)]
enum StampMode {
    Auto,
    Source,
    ImuAnchored,
}

struct RelayState {
    mode: StampMode,
    latest_imu_ns: Option<i64>,
    latest_imu_wall: Option<Instant>,
    imu_count: u64,
    image_count: u64,
    image_without_imu: u64,
    odometry_count: u64,
    started_at: Instant,
}

impl RelayState {
    fn new(mode: StampMode) -> Self {
        Self {
            mode,
            latest_imu_ns: None,
            latest_imu_wall: None,
            imu_count: 0,
            image_count: 0,
            image_without_imu: 0,
            odometry_count: 0,
            started_at: Instant::now(),
        }
    }

    fn report(&self) {
        let elapsed = self.started_at.elapsed().as_secs_f64().max(1e-9);
        log::info!(
            "relay: imu={} ({:.1}Hz) image={} ({:.1}Hz) image_without_imu={} odometry={}",
            self.imu_count,
            self.imu_count as f64 / elapsed,
            self.image_count,
            self.image_count as f64 / elapsed,
            self.image_without_imu,
            self.odometry_count,
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "echo_li_bag_time_relay", "")?;

    fn get_str(node: &r2r::Node, name: &str, default: &str) -> String {
        let params = node.params.lock().unwrap();
        match params.get(name).map(|p| &p.value) {
            Some(r2r::ParameterValue::String(s)) => s.clone(),
            _ => default.to_string(),
        }
    }

    let source_imu = get_str(&node, "source_imu_topic", "/voxl/raw_imu");
    let source_image = get_str(&node, "source_image_topic", "/tracking_front/decoded");
    let output_imu = get_str(&node, "output_imu_topic", "/echo_li_test/imu");
    let output_image = get_str(&node, "output_image_topic", "/echo_li_test/image");
    let odom_topic = get_str(&node, "odometry_topic", "/echo_li/odometry");
    let stamp_mode_str = get_str(&node, "image_stamp_mode", "auto");

    let initial_mode = match stamp_mode_str.as_str() {
        "source" => StampMode::Source,
        "imu_anchored" => StampMode::ImuAnchored,
        "auto" | _ => StampMode::Auto,
    };

    let sensor_qos = best_effort_qos(2000);
    let image_qos = best_effort_qos(5);

    let imu_pub = node.create_publisher::<Imu>(&output_imu, sensor_qos.clone())?;
    let image_pub = node.create_publisher::<Image>(&output_image, image_qos.clone())?;

    let mut imu_sub = node.subscribe::<Imu>(&source_imu, sensor_qos)?;
    let mut image_sub = node.subscribe::<Image>(&source_image, image_qos)?;
    let mut odom_sub =
        node.subscribe::<r2r::nav_msgs::msg::Odometry>(&odom_topic, r2r::QosProfile::default())?;
    let mut timer = node.create_wall_timer(std::time::Duration::from_secs(5))?;

    let _spin = tokio::task::spawn_blocking(move || {
        loop {
            node.spin_once(std::time::Duration::from_millis(10));
        }
    });

    let mut state = RelayState::new(initial_mode);

    loop {
        tokio::select! {
            Some(msg) = imu_sub.next() => {
                state.latest_imu_ns = Some(stamp_ns(&msg.header.stamp));
                state.latest_imu_wall = Some(Instant::now());
                state.imu_count += 1;
                imu_pub.publish(&msg)?;
            }
            Some(mut msg) = image_sub.next() => {
                let Some(latest_imu_ns) = state.latest_imu_ns else {
                    state.image_without_imu += 1;
                    continue;
                };
                if state.mode == StampMode::Auto {
                    let gap_s = (stamp_ns(&msg.header.stamp) - latest_imu_ns) as f64
                        / NSEC_PER_SEC as f64;
                    state.mode = if gap_s.abs() < 1.0 {
                        StampMode::Source
                    } else {
                        StampMode::ImuAnchored
                    };
                    let mode_name = if state.mode == StampMode::Source {
                        "source"
                    } else {
                        "imu_anchored"
                    };
                    log::info!(
                        "image_stamp_mode=auto -> {} (camera stamp - latest IMU stamp = {:+.3}s)",
                        mode_name, gap_s
                    );
                }
                if state.mode == StampMode::ImuAnchored {
                    if let Some(wall) = state.latest_imu_wall {
                        let elapsed_ns = wall.elapsed().as_nanos() as i64;
                        let new_ns = latest_imu_ns + elapsed_ns;
                        msg.header.stamp = echo_li_ros2::to_stamp(new_ns);
                    }
                }
                state.image_count += 1;
                image_pub.publish(&msg)?;
            }
            Some(_) = odom_sub.next() => {
                state.odometry_count += 1;
            }
            _ = timer.tick() => {
                state.report();
            }
        }
    }
}
