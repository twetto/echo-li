//! Flight-height gate for the occupancy map.
//!
//! Take-off and landing ruin a local occupancy map: the camera sweeps the
//! floor at close range while descending, the depth there is poor, and the
//! misses carve out what the flight built. The gate lets depth into the map
//! only while the vehicle holds its programmed flight height.
//!
//! Two height sources, each checked against the band
//! `flight_height_m ± tolerance_m`:
//! - the VIO: odom-frame z of the body minus `vio_ground_z` (the odom origin
//!   is where the filter initialised, i.e. on the ground before take-off);
//! - the rangefinder, while its latest reading is younger than
//!   `range_timeout_s`.
//!
//! By default the gate is closed as soon as *either* available source is
//! outside the band. With `range_overrides_vio`, a fresh rangefinder reading
//! decides alone and the VIO votes only while the rangefinder is absent or
//! stale: the VIO's height drifts by tens of centimetres over a flight, the
//! rangefinder measures it. A source that is absent (no rangefinder, or a
//! stale reading) does not vote. The caller supplies the time for both calls in one clock of its
//! choice (the node uses arrival time, the offline harness bag time).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FlightHeightGateSettings {
    pub enabled: bool,
    /// Programmed flight height above the take-off point, metres.
    pub flight_height_m: f64,
    /// Half-width of the accepted band around `flight_height_m`, metres.
    pub tolerance_m: f64,
    /// Odom-frame z that corresponds to the ground (0 when the VIO
    /// initialises on the ground).
    pub vio_ground_z: f64,
    /// Use the VIO height as a source.
    pub use_vio: bool,
    /// Use the rangefinder as a source (when readings arrive).
    pub use_range: bool,
    /// A rangefinder reading older than this does not vote, seconds.
    pub range_timeout_s: f64,
    /// A fresh rangefinder reading decides alone; the VIO votes only when
    /// the rangefinder is absent or stale.
    pub range_overrides_vio: bool,
}

impl Default for FlightHeightGateSettings {
    fn default() -> Self {
        // Neutral: off. The programmed height belongs in the params file.
        Self {
            enabled: false,
            flight_height_m: 0.0,
            tolerance_m: 0.0,
            vio_ground_z: 0.0,
            use_vio: true,
            use_range: true,
            range_timeout_s: 1.0,
            range_overrides_vio: false,
        }
    }
}

/// One evaluation of the gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateDecision {
    pub open: bool,
    /// VIO height used (None when the VIO does not vote).
    pub vio_height: Option<f64>,
    /// Rangefinder height used (None when absent or stale).
    pub range_height: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct FlightHeightGate {
    settings: FlightHeightGateSettings,
    last_range: Option<(f64, f64)>,
    last_open: Option<bool>,
}

impl FlightHeightGate {
    pub fn new(settings: FlightHeightGateSettings) -> anyhow::Result<Self> {
        if settings.enabled {
            anyhow::ensure!(
                settings.flight_height_m > 0.0,
                "FlightHeightGate.flight_height_m must be set (> 0) when the gate is enabled"
            );
            anyhow::ensure!(
                settings.tolerance_m > 0.0,
                "FlightHeightGate.tolerance_m must be set (> 0) when the gate is enabled"
            );
            anyhow::ensure!(
                settings.use_vio || settings.use_range,
                "FlightHeightGate needs at least one of use_vio / use_range"
            );
        }
        Ok(Self {
            settings,
            last_range: None,
            last_open: None,
        })
    }

    pub fn settings(&self) -> &FlightHeightGateSettings {
        &self.settings
    }

    pub fn band(&self) -> (f64, f64) {
        let s = &self.settings;
        (
            s.flight_height_m - s.tolerance_m,
            s.flight_height_m + s.tolerance_m,
        )
    }

    /// Record a rangefinder reading (metres, already checked against the
    /// sensor's own valid interval by the caller).
    pub fn update_range(&mut self, t_s: f64, range_m: f64) {
        if range_m.is_finite() {
            self.last_range = Some((t_s, range_m));
        }
    }

    /// Decide for a frame at `t_s` with the VIO body z `vio_z` (odom frame).
    pub fn evaluate(&mut self, t_s: f64, vio_z: f64) -> GateDecision {
        let s = &self.settings;
        if !s.enabled {
            return GateDecision {
                open: true,
                vio_height: None,
                range_height: None,
            };
        }
        let (lo, hi) = self.band();
        let inside = |h: f64| h >= lo && h <= hi;
        let vio_height = (s.use_vio && vio_z.is_finite()).then(|| vio_z - s.vio_ground_z);
        let range_height = if s.use_range {
            self.last_range
                .filter(|&(t, _)| (t_s - t).abs() <= s.range_timeout_s)
                .map(|(_, r)| r)
        } else {
            None
        };
        let open = match (range_height, s.range_overrides_vio) {
            (Some(r), true) => inside(r),
            _ => vio_height.map_or(true, inside) && range_height.map_or(true, inside),
        };
        self.last_open = Some(open);
        GateDecision {
            open,
            vio_height,
            range_height,
        }
    }

    /// Whether the previous evaluation left the gate open (None before the
    /// first evaluation). For logging transitions.
    pub fn last_open(&self) -> Option<bool> {
        self.last_open
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> FlightHeightGate {
        FlightHeightGate::new(FlightHeightGateSettings {
            enabled: true,
            flight_height_m: 1.0,
            tolerance_m: 0.3,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn disabled_gate_is_always_open() {
        let mut g = FlightHeightGate::new(FlightHeightGateSettings::default()).unwrap();
        assert!(g.evaluate(0.0, -5.0).open);
    }

    #[test]
    fn enabled_gate_requires_a_height() {
        let s = FlightHeightGateSettings {
            enabled: true,
            ..Default::default()
        };
        assert!(FlightHeightGate::new(s).is_err());
    }

    #[test]
    fn vio_alone_opens_inside_the_band_only() {
        let mut g = gate();
        assert!(!g.evaluate(0.0, 0.05).open); // on the ground
        assert!(g.evaluate(1.0, 0.9).open); // cruising
        assert!(!g.evaluate(2.0, 1.5).open); // climbed out of the band
        assert!(!g.evaluate(3.0, 0.5).open); // landing
    }

    #[test]
    fn either_source_out_of_band_closes_the_gate() {
        let mut g = gate();
        g.update_range(10.0, 0.4);
        let d = g.evaluate(10.2, 0.95);
        assert!(!d.open);
        assert_eq!(d.range_height, Some(0.4));
        g.update_range(10.5, 1.0);
        assert!(g.evaluate(10.6, 0.95).open);
        assert!(!g.evaluate(10.7, 0.5).open); // VIO says landing, range still fine
    }

    #[test]
    fn fresh_range_overrides_a_drifting_vio() {
        let mut g = FlightHeightGate::new(FlightHeightGateSettings {
            enabled: true,
            flight_height_m: 1.0,
            tolerance_m: 0.3,
            range_overrides_vio: true,
            ..Default::default()
        })
        .unwrap();
        g.update_range(0.0, 1.0);
        assert!(g.evaluate(0.2, 0.66).open); // VIO drifted low, rangefinder says 1.0 m
        assert!(g.evaluate(0.3, 1.32).open); // VIO drifted high
        g.update_range(0.5, 0.5);
        assert!(!g.evaluate(0.6, 1.0).open); // landing: the rangefinder closes it
        // stale rangefinder: the VIO decides again
        assert!(!g.evaluate(5.0, 0.5).open);
        assert!(g.evaluate(5.1, 1.0).open);
    }

    #[test]
    fn stale_range_does_not_vote() {
        let mut g = gate();
        g.update_range(0.0, 0.1);
        assert!(!g.evaluate(0.5, 1.0).open);
        let d = g.evaluate(1.6, 1.0);
        assert!(d.open);
        assert_eq!(d.range_height, None);
    }

    #[test]
    fn ground_reference_shifts_the_vio_height() {
        let mut g = FlightHeightGate::new(FlightHeightGateSettings {
            enabled: true,
            flight_height_m: 1.0,
            tolerance_m: 0.3,
            vio_ground_z: -1.0, // filter initialised 1 m above the ground
            ..Default::default()
        })
        .unwrap();
        assert!(g.evaluate(0.0, 0.0).open);
        assert!(!g.evaluate(0.0, -0.9).open);
    }
}
