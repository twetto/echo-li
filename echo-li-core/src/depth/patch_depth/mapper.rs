use super::*;

impl PatchDepthMapper {
    pub fn update(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        seed_coordinates: PatchDepthSeedCoordinates,
        frame: FrameProducts,
    ) -> Option<PatchDepthOutput> {
        if seed_coordinates != self.expected_seed_coordinates() {
            return None;
        }
        let seeds = self.gather_seeds(sparse_filter, measurement);
        self.update_with_priors(frame, &seeds, None, 0.0)
    }

    pub fn update_with_priors(
        &mut self,
        frame: FrameProducts,
        seeds: &[SparseDepthPrior],
        p_vv: Option<&Matrix3<f64>>,
        dt: f64,
    ) -> Option<PatchDepthOutput> {
        self.update_with_priors_and_pose_covariances(frame, seeds, p_vv, None, dt)
    }

    pub fn update_with_priors_and_pose_covariances(
        &mut self,
        frame: FrameProducts,
        seeds: &[SparseDepthPrior],
        p_vv: Option<&Matrix3<f64>>,
        p_ww: Option<&Matrix3<f64>>,
        dt: f64,
    ) -> Option<PatchDepthOutput> {
        if frame.width != self.width
            || frame.height != self.height
            || frame.gray.len() != self.width * self.height
        {
            return None;
        }
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            return self.update_with_priors_tiled_bearing(frame, seeds, p_vv, p_ww, dt);
        }

        let depth_frame = self.depth_frame_products(frame)?;
        let median_depth = median_seed_depth(seeds).unwrap_or(self.settings.max_depth);
        let selected = self.select_keyframe(&depth_frame.frame.pose_t_wc, median_depth);
        self.manage_keyframes(&depth_frame, median_depth);
        let (ref_keyframe, t_ref_curr) = selected?;
        let warp_uncertainty = compute_warp_uncertainty(
            &self.intrinsics,
            &t_ref_curr,
            p_vv,
            p_ww,
            self.settings.pose_angular_velocity_var,
            dt,
            median_depth,
        );
        Some(self.solve(
            &depth_frame,
            &ref_keyframe,
            &t_ref_curr,
            seeds,
            warp_uncertainty,
        ))
    }

    pub(super) fn update_with_priors_tiled_bearing(
        &mut self,
        frame: FrameProducts,
        seeds: &[SparseDepthPrior],
        p_vv: Option<&Matrix3<f64>>,
        p_ww: Option<&Matrix3<f64>>,
        dt: f64,
    ) -> Option<PatchDepthOutput> {
        let depth_frame = self.tiled_bearing_frame_products(frame)?;
        let median_depth = median_seed_depth(seeds).unwrap_or(self.settings.max_depth);
        let selected = self.select_tiled_keyframe(&depth_frame.frame.pose_t_wc, median_depth);
        self.manage_tiled_keyframes(&depth_frame, median_depth);
        let (ref_keyframe, t_ref_curr) = selected?;
        let warp_uncertainty = compute_warp_uncertainty(
            &self.intrinsics,
            &t_ref_curr,
            p_vv,
            p_ww,
            self.settings.pose_angular_velocity_var,
            dt,
            median_depth,
        );
        Some(self.solve_tiled_bearing(
            &depth_frame,
            &ref_keyframe,
            &t_ref_curr,
            seeds,
            warp_uncertainty.scalar_sq,
        ))
    }

    /// Set up stereo reference frame support. Builds rectification LUTs that
    /// map cam0-pinhole pixels to cam1 raw pixels, so cam1 images can be used
    /// as reference frames with the existing warp/project pipeline.
    ///
    /// `cam1_model` must use the same resolution as cam0.
    pub fn init_stereo_ref(&mut self, cam1_model: &dyn CameraModel, t_c1_c0: Matrix4<f64>) {
        self.stereo_t_c1_c0 = Some(t_c1_c0);
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            self.stereo_tiled_bearing_levels = Some(Arc::new(build_tiled_bearing_levels(
                cam1_model,
                &self.intrinsics,
                self.width,
                self.height,
                self.settings.scale,
                self.settings.n_pyramid_levels,
                self.settings.tiled_tile_size,
                self.settings.tiled_tile_overlap,
            )));
            return;
        }
        if self.camera_mode != PatchDepthCameraMode::UndistortedPinhole {
            self.stereo_t_c1_c0 = None;
            return;
        }
        let specs = undistort_level_specs(
            self.width,
            self.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let luts: Vec<UndistortLut> = specs
            .iter()
            .map(|spec| {
                UndistortLut::from_options(
                    build_pinhole_to_raw_lut(cam1_model, &self.intrinsics, spec),
                    spec.lw,
                    spec.lh,
                )
            })
            .collect();
        let valid = Arc::new(
            luts.iter()
                .map(UndistortLut::valid_image)
                .collect::<Vec<_>>(),
        );
        self.stereo_undistort_luts = Some(luts);
        self.stereo_valid_pyramid = Some(valid);
    }

    pub fn has_stereo_ref(&self) -> bool {
        self.stereo_undistort_luts.is_some() || self.stereo_tiled_bearing_levels.is_some()
    }

    /// Use cam1 as the reference frame instead of the motion-based keyframe pool.
    pub fn update_with_stereo_ref(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        seed_coordinates: PatchDepthSeedCoordinates,
        frame: FrameProducts,
        cam1_gray: &[u8],
        cam1_width: usize,
        cam1_height: usize,
    ) -> Option<PatchDepthOutput> {
        if seed_coordinates != self.expected_seed_coordinates() {
            return None;
        }
        if self.camera_mode == PatchDepthCameraMode::TiledBearing {
            return self.update_with_stereo_ref_tiled_bearing(
                sparse_filter,
                measurement,
                frame,
                cam1_gray,
                cam1_width,
                cam1_height,
            );
        }
        let stereo_luts = self.stereo_undistort_luts.as_ref()?;
        let t_c1_c0 = self.stereo_t_c1_c0?;

        let raw_cam1_pyr = build_pyramid_from_u8(
            cam1_gray,
            cam1_width,
            cam1_height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let rectified_pyr: Vec<Image<f32>> = raw_cam1_pyr
            .iter()
            .zip(stereo_luts)
            .map(|(raw, lut)| lut.undistort_level(raw))
            .collect();
        let ref_valid_pyr = self.stereo_valid_pyramid.clone();
        let bv_pyr = ref_valid_pyr
            .as_ref()
            .map(|p| build_bilinear_valid_pyramid(p));
        let mut grad_x_pyr = Vec::with_capacity(self.settings.n_pyramid_levels);
        let mut grad_y_pyr = Vec::with_capacity(self.settings.n_pyramid_levels);
        for img in &rectified_pyr {
            let (gx, gy) = gradients(img);
            grad_x_pyr.push(gx);
            grad_y_pyr.push(gy);
        }
        let ref_keyframe = DepthKeyframe {
            frame: Arc::new(FrameProducts {
                frame_id: 0,
                stamp: frame.stamp,
                gray: Vec::new(),
                width: cam1_width,
                height: cam1_height,
                pose_t_wc: Matrix4::identity(),
            }),
            ref_pyramid: Arc::new(rectified_pyr),
            bilinear_valid_pyramid: bv_pyr.map(Arc::new),
            grad_x_pyramid: grad_x_pyr,
            grad_y_pyramid: grad_y_pyr,
        };

        let seeds = self.gather_seeds(sparse_filter, measurement);
        let depth_frame = self.depth_frame_products(frame)?;
        let median_depth = median_seed_depth(&seeds).unwrap_or(self.settings.max_depth);
        self.manage_keyframes(&depth_frame, median_depth);

        Some(self.solve(
            &depth_frame,
            &ref_keyframe,
            &t_c1_c0,
            &seeds,
            WarpUncertainty::scalar(0.0),
        ))
    }

    pub(super) fn update_with_stereo_ref_tiled_bearing(
        &mut self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
        frame: FrameProducts,
        cam1_gray: &[u8],
        cam1_width: usize,
        cam1_height: usize,
    ) -> Option<PatchDepthOutput> {
        if cam1_width != self.width
            || cam1_height != self.height
            || cam1_gray.len() != self.width * self.height
        {
            return None;
        }
        let stereo_layout = Arc::clone(self.stereo_tiled_bearing_levels.as_ref()?);
        let t_c1_c0 = self.stereo_t_c1_c0?;
        let seeds = self.gather_seeds(sparse_filter, measurement);
        let depth_frame = self.tiled_bearing_frame_products(frame)?;

        let raw_cam1_pyramid = Arc::new(build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            cam1_gray,
            cam1_width,
            cam1_height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        ));
        let cam1_levels =
            build_tiled_bearing_frame_levels(stereo_layout.as_ref(), raw_cam1_pyramid.as_ref());
        let cam1_frame_products = TiledBearingFrameProducts {
            frame: Arc::new(FrameProducts {
                frame_id: 0,
                stamp: depth_frame.frame.stamp,
                gray: Vec::new(),
                width: cam1_width,
                height: cam1_height,
                pose_t_wc: Matrix4::identity(),
            }),
            raw_pyramid: raw_cam1_pyramid,
            levels: cam1_levels,
        };
        let ref_keyframe = self.make_tiled_bearing_keyframe(&cam1_frame_products);
        let median_depth = median_seed_depth(&seeds).unwrap_or(self.settings.max_depth);
        self.manage_tiled_keyframes(&depth_frame, median_depth);

        Some(self.solve_tiled_bearing(&depth_frame, &ref_keyframe, &t_c1_c0, &seeds, 0.0))
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    pub(super) fn depth_frame_products(
        &mut self,
        frame: FrameProducts,
    ) -> Option<DepthFrameProducts> {
        // Build the pyramid on the raw image, then undistort each kept level.
        // Undistorting the small downsampled levels instead of the full-res
        // frame is ~16x less work; the anti-alias blur happens in raw space.
        let raw_pyramid = build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        );
        let (pyramid, valid_pyramid) = match self.camera_mode {
            PatchDepthCameraMode::RawDistorted | PatchDepthCameraMode::PerPatchBearing => {
                (raw_pyramid, None)
            }
            PatchDepthCameraMode::UndistortedPinhole | PatchDepthCameraMode::TiledBearing => {
                let luts = self.undistort_luts.as_ref()?;
                let undistorted: Vec<Image<f32>> = raw_pyramid
                    .iter()
                    .zip(luts)
                    .map(|(raw, lut)| lut.undistort_level(raw))
                    .collect();
                (undistorted, self.valid_pyramid.clone())
            }
        };
        Some(DepthFrameProducts {
            frame: Arc::new(frame),
            pyramid: Arc::new(pyramid),
            valid_pyramid,
        })
    }

    #[allow(dead_code)]
    pub(super) fn tiled_bearing_frame_products(
        &mut self,
        frame: FrameProducts,
    ) -> Option<TiledBearingFrameProducts> {
        let layout = Arc::clone(self.tiled_bearing_levels.as_ref()?);
        let raw_pyramid = Arc::new(build_working_pyramid(
            &mut self.pyramid_work,
            &mut self.pyramid_scratch,
            &frame.gray,
            frame.width,
            frame.height,
            self.settings.scale,
            self.settings.n_pyramid_levels,
        ));
        let levels = build_tiled_bearing_frame_levels(layout.as_ref(), raw_pyramid.as_ref());
        Some(TiledBearingFrameProducts {
            frame: Arc::new(frame),
            raw_pyramid,
            levels,
        })
    }

    #[allow(dead_code)]
    pub(super) fn make_tiled_bearing_keyframe(
        &self,
        frame_products: &TiledBearingFrameProducts,
    ) -> TiledBearingKeyframe {
        let levels = frame_products
            .levels
            .iter()
            .map(|level| {
                let tiles = level
                    .tiles
                    .iter()
                    .map(|tile| {
                        let (grad_x, grad_y) = gradients(&tile.image);
                        TiledBearingKeyframeTile {
                            tile: tile.tile.clone(),
                            ref_image: tile.image.clone(),
                            bilinear_valid: bilinear_valid_image_from_mask(&tile.valid),
                            grad_x,
                            grad_y,
                        }
                    })
                    .collect();
                TiledBearingKeyframeLevel {
                    level: level.level,
                    width: level.width,
                    height: level.height,
                    tiles,
                }
            })
            .collect();
        TiledBearingKeyframe {
            frame: Arc::clone(&frame_products.frame),
            levels,
        }
    }

    pub(super) fn select_tiled_keyframe(
        &self,
        t_wc: &Matrix4<f64>,
        median_depth: f64,
    ) -> Option<(TiledBearingKeyframe, Matrix4<f64>)> {
        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let max_bl = self.settings.max_baseline_ratio * median_depth;
        let mut best: Option<(TiledBearingKeyframe, Matrix4<f64>, f64)> = None;

        for keyframe in &self.tiled_keyframes {
            let t_ref_curr = keyframe
                .frame
                .pose_t_wc
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
                * t_wc;
            let baseline = t_ref_curr.fixed_view::<3, 1>(0, 3).norm();
            if baseline >= min_bl
                && baseline <= max_bl
                && best.as_ref().map(|(_, _, b)| baseline > *b).unwrap_or(true)
            {
                best = Some((keyframe.clone(), t_ref_curr, baseline));
            }
        }

        best.map(|(kf, t, _)| (kf, t))
    }

    pub(super) fn manage_tiled_keyframes(
        &mut self,
        depth_frame: &TiledBearingFrameProducts,
        median_depth: f64,
    ) {
        if self.tiled_keyframes.len() < 2 {
            self.tiled_keyframes
                .push(self.make_tiled_bearing_keyframe(depth_frame));
            return;
        }

        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let newest = &self.tiled_keyframes[self.tiled_keyframes.len() - 1];
        let t_new_curr = newest
            .frame
            .pose_t_wc
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
            * depth_frame.frame.pose_t_wc;
        let baseline = t_new_curr.fixed_view::<3, 1>(0, 3).norm();
        if baseline >= min_bl {
            self.tiled_keyframes.remove(0);
            self.tiled_keyframes
                .push(self.make_tiled_bearing_keyframe(depth_frame));
        }
    }

    pub(super) fn gather_seeds(
        &self,
        sparse_filter: &Sparse3DFilter,
        measurement: &VisionMeasurement,
    ) -> Vec<SparseDepthPrior> {
        let intr_orig = ScaledIntrinsics {
            scale_from_original: 1.0,
        };
        let is_tiled = self.camera_mode == PatchDepthCameraMode::TiledBearing;
        let mut fids: Vec<u64> = measurement.cam_coordinates.keys().copied().collect();
        fids.sort_unstable();
        let mut seeds = Vec::new();
        for fid in fids {
            let uv_f32 = &measurement.cam_coordinates[&fid];
            let (depth, depth_var) = if is_tiled {
                sparse_filter.query_range(fid)
            } else {
                sparse_filter.query(fid)
            };
            if depth <= 0.0
                || depth < self.settings.min_depth
                || depth > self.settings.max_depth
                || !depth_var.is_finite()
            {
                continue;
            }
            let uv = Vector2::new(uv_f32[0] as f64, uv_f32[1] as f64);
            // range_per_z = ‖bearing‖; for tiled seeds (rho = 1/range) this is 1.0.
            let range_per_z = if is_tiled {
                1.0
            } else {
                self.bearing_for_scaled_pixel(uv[0], uv[1], &intr_orig)
                    .map(|b| b.norm())
                    .unwrap_or(1.0)
            };
            let rho = 1.0 / depth;
            let rho_var = depth_var / depth.powi(4);
            let eta = rho_to_eta(rho, range_per_z);
            let eta_var = rho_var_to_eta_var(rho, rho_var);
            if eta.is_finite() && eta_var.is_finite() && eta_var > 0.0 {
                seeds.push(SparseDepthPrior { uv, eta, eta_var });
            }
        }
        seeds
    }

    pub(super) fn select_keyframe(
        &self,
        t_wc: &Matrix4<f64>,
        median_depth: f64,
    ) -> Option<(DepthKeyframe, Matrix4<f64>)> {
        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let max_bl = self.settings.max_baseline_ratio * median_depth;
        let mut best: Option<(DepthKeyframe, Matrix4<f64>, f64)> = None;

        for keyframe in &self.keyframes {
            let t_ref_curr = keyframe
                .frame
                .pose_t_wc
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
                * t_wc;
            let baseline = t_ref_curr.fixed_view::<3, 1>(0, 3).norm();
            if baseline >= min_bl
                && baseline <= max_bl
                && best.as_ref().map(|(_, _, b)| baseline > *b).unwrap_or(true)
            {
                best = Some((keyframe.clone(), t_ref_curr, baseline));
            }
        }

        best.map(|(kf, t, _)| (kf, t))
    }

    pub(super) fn manage_keyframes(&mut self, depth_frame: &DepthFrameProducts, median_depth: f64) {
        if self.keyframes.len() < 2 {
            self.keyframes.push(self.make_keyframe(depth_frame));
            return;
        }

        let min_bl = self.settings.min_baseline_ratio * median_depth;
        let newest = &self.keyframes[self.keyframes.len() - 1];
        let t_new_curr = newest
            .frame
            .pose_t_wc
            .try_inverse()
            .unwrap_or_else(Matrix4::identity)
            * depth_frame.frame.pose_t_wc;
        let baseline = t_new_curr.fixed_view::<3, 1>(0, 3).norm();
        if baseline >= min_bl {
            self.keyframes.remove(0);
            self.keyframes.push(self.make_keyframe(depth_frame));
        }
    }

    pub(super) fn make_keyframe(&self, depth_frame: &DepthFrameProducts) -> DepthKeyframe {
        let bilinear_valid_pyramid = depth_frame
            .valid_pyramid
            .as_ref()
            .map(|pyramid| build_bilinear_valid_pyramid(pyramid));
        let mut grad_x_pyramid = Vec::with_capacity(self.settings.n_pyramid_levels);
        let mut grad_y_pyramid = Vec::with_capacity(self.settings.n_pyramid_levels);
        for img in depth_frame.pyramid.iter() {
            let (gx, gy) = gradients(img);
            grad_x_pyramid.push(gx);
            grad_y_pyramid.push(gy);
        }
        DepthKeyframe {
            frame: Arc::clone(&depth_frame.frame),
            ref_pyramid: Arc::clone(&depth_frame.pyramid),
            bilinear_valid_pyramid: bilinear_valid_pyramid.map(Arc::new),
            grad_x_pyramid,
            grad_y_pyramid,
        }
    }
}
