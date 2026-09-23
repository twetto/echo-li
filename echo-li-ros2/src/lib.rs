pub const NSEC_PER_SEC: i64 = 1_000_000_000;

pub fn stamp_ns(stamp: &r2r::builtin_interfaces::msg::Time) -> i64 {
    stamp.sec as i64 * NSEC_PER_SEC + stamp.nanosec as i64
}

pub fn to_stamp(ns: i64) -> r2r::builtin_interfaces::msg::Time {
    r2r::builtin_interfaces::msg::Time {
        sec: (ns / NSEC_PER_SEC) as i32,
        nanosec: (ns % NSEC_PER_SEC) as u32,
    }
}

/// Build an unordered XYZI `PointCloud2` from packed little-endian f32 quads.
/// `data` must hold exactly `n` points of x, y, z, intensity.
pub fn xyzi_cloud(
    stamp: r2r::builtin_interfaces::msg::Time,
    frame_id: &str,
    data: Vec<u8>,
    n: u32,
) -> r2r::sensor_msgs::msg::PointCloud2 {
    const POINT_STEP: u32 = 16;
    let field = |name: &str, offset: u32| r2r::sensor_msgs::msg::PointField {
        name: name.into(),
        offset,
        datatype: 7, // FLOAT32
        count: 1,
    };
    r2r::sensor_msgs::msg::PointCloud2 {
        header: r2r::std_msgs::msg::Header {
            stamp,
            frame_id: frame_id.to_string(),
        },
        height: 1,
        width: n,
        fields: vec![
            field("x", 0),
            field("y", 4),
            field("z", 8),
            field("intensity", 12),
        ],
        is_bigendian: false,
        point_step: POINT_STEP,
        row_step: POINT_STEP * n,
        data,
        is_dense: true,
    }
}

pub fn quat_to_se3(position: &[f64; 3], quaternion: &[f64; 4]) -> nalgebra::Matrix4<f64> {
    let [x, y, z, w] = *quaternion;
    let mut m = nalgebra::Matrix4::identity();
    m[(0, 0)] = 1.0 - 2.0 * (y * y + z * z);
    m[(0, 1)] = 2.0 * (x * y - z * w);
    m[(0, 2)] = 2.0 * (x * z + y * w);
    m[(1, 0)] = 2.0 * (x * y + z * w);
    m[(1, 1)] = 1.0 - 2.0 * (x * x + z * z);
    m[(1, 2)] = 2.0 * (y * z - x * w);
    m[(2, 0)] = 2.0 * (x * z - y * w);
    m[(2, 1)] = 2.0 * (y * z + x * w);
    m[(2, 2)] = 1.0 - 2.0 * (x * x + y * y);
    m[(0, 3)] = position[0];
    m[(1, 3)] = position[1];
    m[(2, 3)] = position[2];
    m
}

pub fn best_effort_qos(depth: usize) -> r2r::QosProfile {
    r2r::QosProfile {
        depth,
        ..r2r::QosProfile::sensor_data()
    }
}
