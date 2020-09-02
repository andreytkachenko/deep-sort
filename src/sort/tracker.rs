use ndarray::prelude::*;
use std::collections::HashSet;
use crate::sort::{Track, Detection, KalmanFilter, DistanceMetric};

/// This is the multi-target tracker.
///
///     Parameters
///     ----------
///     metric : nn_matching.NearestNeighborDistanceMetric
///         A distance metric for measurement-to-track association.
///     max_age : int
///         Maximum number of missed misses before a track is deleted.
///     n_init : int
///         Number of consecutive detections before the track is confirmed. The
///         track state is set to `Deleted` if a miss occurs within the first
///         `n_init` frames.
///
///     Attributes
///     ----------
///     metric : nn_matching.NearestNeighborDistanceMetric
///         The distance metric used for measurement to track association.
///     max_age : int
///         Maximum number of missed misses before a track is deleted.
///     n_init : int
///         Number of frames that a track remains in initialization phase.
///     kf : kalman_filter.KalmanFilter
///         A Kalman filter to filter target trajectories in image space.
///     tracks : List[Track]
///         The list of active tracks at the current time step.
///
#[derive(Clone)]
pub struct Tracker<M: DistanceMetric> {
    metric: M,
    max_iou_distance: f32,
    max_age: i32,
    n_init: i32,
    kf: KalmanFilter,
    tracks: Vec<Track>,
    next_id: i32,
}

impl<M: DistanceMetric> Tracker<M> {
    pub fn new(metric: M, max_iou_distance: f32 /*=0.7*/, max_age: i32/*=70*/, n_init: i32/*=3*/) -> Self {
        Self {
            metric,
            max_iou_distance,
            max_age,
            n_init,
            kf: Default::default(),
            next_id: 1,
            tracks: Vec::new(),
        }
    }

    #[inline]
    pub fn tracks(&self) -> &[Track] {
        self.tracks.as_slice()
    }

    ///
    /// Propagate track state distributions one time step forward.
    ///
    /// This function should be called once every time step, before `update`.
    ///
    pub fn predict(&mut self) {
        for track in &mut self.tracks {
            track.predict(&mut self.kf);
        }
    }

    /// Perform measurement update and track management.
    ///
    ///     Parameters
    ///     ----------
    ///     detections : List[deep_sort.detection.Detection]
    ///         A list of detections at the current time step.
    ///
    pub fn update(&mut self, detections: &[Detection]) {
        let (matches, unmatched_tracks, unmatched_detections) = self.do_match(&detections);

        for (track_idx, detection_idx) in matches {
            self.tracks[track_idx].update(
                &mut self.kf, &detections[detection_idx]);
        }

        for track_idx in unmatched_tracks {
            self.tracks[track_idx].mark_missed();
        }

        for detection_idx in unmatched_detections {
            self.initiate_track(&detections[detection_idx]);
        }

        self.tracks.retain(|t|!t.is_deleted());

        let (
            mut features,
            mut targets,
            mut active_targets) = (vec![], vec![], vec![]);

        for track in &mut self.tracks {
            if !track.is_confirmed() {
                continue;
            }

            active_targets.push(track.track_id);

            for feature in track.features.iter() {
                targets.push(track.track_id);
                features.push(feature.clone());
            }
        }

        self.metric.partial_fit(
            features,
            targets,
            active_targets
        )
    }

    fn do_match(&mut self, detections: &[Detection]) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
        let gated_metric = |tracks: &[Track], dets: &[Detection], track_indices: &[usize], detection_indices: &[usize]| {
            let mut features = unsafe { Array2::uninitialized((detection_indices.len(), dets[0].feature.as_ref().unwrap().len())) };

            for (idx, mut axis) in features.axis_iter_mut(Axis(0)).enumerate() {
                let index = dets[detection_indices[idx]].feature.as_ref().unwrap();
                axis.assign(index);
            }

            let targets: Vec<_> = track_indices.iter().map(|&i|tracks[i].track_id).collect();
            let mut cost_matrix = self.metric.distance(features.view(), targets);

            crate::sort::linear_assignment::gate_cost_matrix(
                &self.kf,
                cost_matrix.view_mut(),
                &tracks,
                &dets,
                &track_indices,
                detection_indices,
                None,
                None
            );

            cost_matrix
        };

        // Split track set into confirmed and unconfirmed tracks.
        let (mut confirmed_tracks, mut unconfirmed_tracks) = (vec![], vec![]);
        for (i, t) in self.tracks.iter().enumerate() {
            if t.is_confirmed() {
                confirmed_tracks.push(i);
            } else {
                unconfirmed_tracks.push(i);
            }
        }

        let matching_threshold = self.metric.matching_threshold();
        let max_age = self.max_age;

        // Associate confirmed tracks using appearance features.
        let (matches_a, unmatched_tracks_a, unmatched_detections) =
            crate::sort::linear_assignment::matching_cascade(
                &gated_metric,
                matching_threshold,
                max_age,
                &self.tracks,
                detections,
                Some(confirmed_tracks),
                None);

        // Associate remaining tracks together with unconfirmed tracks using IOU.
        let (iou_track_candidates, unmatched_tracks_a): (Vec<_>, Vec<_>) = unmatched_tracks_a
            .into_iter()
            .partition(|&k|self.tracks[k].time_since_update == 1);

        let iou_track_candidates = [unconfirmed_tracks.as_slice(), iou_track_candidates.as_slice()].concat();

        let (matches_b, unmatched_tracks_b, unmatched_detections) =
            crate::sort::linear_assignment::min_cost_matching(
                &crate::sort::iou_matching::iou_cost,
                self.max_iou_distance,
                &self.tracks,
                &detections,
                Some(iou_track_candidates),
                Some(unmatched_detections));

        let matches = [matches_a, matches_b].concat();
        let unmatched_tracks: HashSet<_> = unmatched_tracks_a
            .into_iter()
            .chain(unmatched_tracks_b.into_iter())
            .collect();

        (matches, unmatched_tracks.into_iter().collect(), unmatched_detections)
    }

    fn initiate_track(&mut self, detection: &Detection) {
        let (mean, covariance) = self.kf.initiate(detection.bbox.as_xyah());

        self.tracks.push(Track::new(
            mean,
            covariance,
            self.next_id,
            self.n_init,
            self.max_age,
            detection.feature.clone()
        ));

        self.next_id += 1;
    }
}
