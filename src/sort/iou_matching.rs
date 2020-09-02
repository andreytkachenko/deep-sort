use ndarray::prelude::*;

use crate::sort::linear_assignment::INFTY_COST;
use crate::sort::{Track, Detection, BBox, Ltwh};

/// Computer intersection over union.
/// Parameters
/// ----------
/// bbox : ndarray
///     A bounding box in format `(top left x, top left y, width, height)`.
/// candidates : ndarray
///     A matrix of candidate bounding boxes (one per row) in the same format
///     as `bbox`.
/// Returns
/// -------
/// ndarray
///     The intersection over union in [0, 1] between the `bbox` and each
///     candidate. A higher score means a larger fraction of the `bbox` is
///     occluded by the candidate.
pub fn iou(bbox: &BBox<Ltwh>, candidates: &[BBox<Ltwh>]) -> Array1<f32> {
    let bbox_area = bbox.width() * bbox.height();

    candidates
        .iter()
        .map(|c_ltwh| {
            let b1 = bbox.as_ltrb();
            let b2 = c_ltwh.as_ltrb();

            let i_xmin = b1.left().max(b2.left());
            let i_ymin = b1.top().max(b2.top());

            let i_xmax = b1.right().min(b2.right());
            let i_ymax = b1.bottom().min(b2.bottom());
            
            let intersection_area = ((i_xmax - i_xmin).max(0.0) * (i_ymax - i_ymin).max(0.0));
            let candidate_area = c_ltwh.width() * c_ltwh.height();

            intersection_area / (bbox_area + candidate_area - intersection_area)
        })
        .collect()
}

///
/// An intersection over union distance metric.
/// Parameters
/// ----------
/// tracks : List[deep_sort.track.Track]
///     A list of tracks.
/// detections : List[deep_sort.detection.Detection]
///     A list of detections.
/// track_indices : Optional[List[int]]
///     A list of indices to tracks that should be matched. Defaults to
///     all `tracks`.
/// detection_indices : Optional[List[int]]
///     A list of indices to detections that should be matched. Defaults
///     to all `detections`.
/// Returns
/// -------
/// ndarray
///     Returns a cost matrix of shape
///     len(track_indices), len(detection_indices) where entry (i, j) is
///     `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
///
pub fn iou_cost(tracks: &[Track], detections: &[Detection], track_indices: &[usize], detection_indices: &[usize]) -> Array2<f32> {
    let track_n = track_indices.len();
    let det_n = detection_indices.len();
    let n = track_n.max(det_n);

    let mut cost_matrix = Array2::from_elem((n, n), 1.0);

    for (row, &track_idx) in track_indices.iter().enumerate() {
        let track = &tracks[track_idx];

        if track.time_since_update > 1 {
            continue;
        } 

        let bbox = track.bbox().as_ltwh();
        let candidates: Vec<_> = detection_indices
            .iter()
            .map(|&i| detections[i].bbox.clone())
            .collect();


        cost_matrix.slice_mut(s![row, ..det_n]).assign(&(1.0 - iou(&bbox, &candidates)));
    }

    cost_matrix
}