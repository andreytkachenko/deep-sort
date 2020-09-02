use std::collections::HashSet;

use ndarray::prelude::*;
use crate::sort::{Track, Detection, KalmanFilter};

pub const INFTY_COST: f32 = 1e+5;


/// Solve linear assignment problem.
///
/// Parameters
/// ----------
/// distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
///     The distance metric is given a list of tracks and detections as well as
///     a list of N track indices and M detection indices. The metric should
///     return the NxM dimensional cost matrix, where element (i, j) is the
///     association cost between the i-th track in the given track indices and
///     the j-th detection in the given detection_indices.
/// max_distance : float
///     Gating threshold. Associations with cost larger than this value are
///     disregarded.
/// tracks : List[track.Track]
///     A list of predicted tracks at the current time step.
/// detections : List[detection.Detection]
///     A list of detections at the current time step.
/// track_indices : List[int]
///     List of track indices that maps rows in `cost_matrix` to tracks in
///     `tracks` (see description above).
/// detection_indices : List[int]
///     List of detection indices that maps columns in `cost_matrix` to
///     detections in `detections` (see description above).
/// 
/// Returns
/// -------
/// (List[(int, int)], List[int], List[int])
///     Returns a tuple with the following three entries:
///     * A list of matched track and detection indices.
///     * A list of unmatched track indices.
///     * A list of unmatched detection indices.
/// 
pub fn min_cost_matching<D: Fn(&[Track], &[Detection], &[usize], &[usize]) -> Array2<f32>>(
    distance_metric: &D, 
    max_distance: f32, 
    tracks: &[Track], 
    detections: &[Detection], 
    track_indices: Option<Vec<usize>>, 
    detection_indices: Option<Vec<usize>>
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    let track_indices = track_indices.unwrap_or_else(||(0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(||(0..detections.len()).collect());

    if detection_indices.is_empty() || track_indices.is_empty() {
        return (vec![], track_indices, detection_indices);  // Nothing to match.
    }

    let mut cost_matrix = distance_metric(tracks, detections, &track_indices, &detection_indices);
    cost_matrix.mapv_inplace(|x| if x > max_distance { max_distance + 1.0e-5 } else { x });

    let mut weights = munkres::WeightMatrix::from_row_vec(cost_matrix.nrows(), cost_matrix.iter().copied().collect());
    let indices = munkres::solve_assignment(&mut weights).unwrap();
    
    let (mut matches, mut unmatched_tracks, mut unmatched_detections) = (vec![], vec![], vec![]);
    
    for (idx, &detection_idx) in detection_indices.iter().enumerate() {
        let mut present = false;

        for pos in indices.iter() {
            if idx == pos.column && pos.row < track_indices.len() {
                present = true;
                break;
            }
        }

        if !present {
            unmatched_detections.push(detection_idx);
        }
    }

    for (idx, &track_idx) in track_indices.iter().enumerate() {
        let mut present = false;

        for pos in indices.iter() {
            if idx == pos.row && pos.column < detection_indices.len() {
                present = true;
                break;
            }
        }

        if !present {
            unmatched_tracks.push(track_idx);
        }
    }

    for pos in indices.into_iter() {
        if pos.row < track_indices.len() &&
           pos.column < detection_indices.len() {

            let track_idx = track_indices[pos.row];
            let detection_idx = detection_indices[pos.column];

            if cost_matrix[(pos.row, pos.column)] > max_distance {
                unmatched_tracks.push(track_idx);
                unmatched_detections.push(detection_idx);
            } else {
                matches.push((track_idx, detection_idx))
            }
        }
    }

    (matches, unmatched_tracks, unmatched_detections)
}

/// Run matching cascade.
/// 
/// Parameters
/// ----------
/// distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
///     The distance metric is given a list of tracks and detections as well as
///     a list of N track indices and M detection indices. The metric should
///     return the NxM dimensional cost matrix, where element (i, j) is the
///     association cost between the i-th track in the given track indices and
///     the j-th detection in the given detection indices.
/// max_distance : float
///     Gating threshold. Associations with cost larger than this value are
///     disregarded.
/// cascade_depth: int
///     The cascade depth, should be se to the maximum track age.
/// tracks : List[track.Track]
///     A list of predicted tracks at the current time step.
/// detections : List[detection.Detection]
///     A list of detections at the current time step.
/// track_indices : Optional[List[int]]
///     List of track indices that maps rows in `cost_matrix` to tracks in
///     `tracks` (see description above). Defaults to all tracks.
/// detection_indices : Optional[List[int]]
///     List of detection indices that maps columns in `cost_matrix` to
///     detections in `detections` (see description above). Defaults to all
///     detections.
/// 
/// Returns
/// -------
/// (List[(int, int)], List[int], List[int])
///     Returns a tuple with the following three entries:
///     * A list of matched track and detection indices.
///     * A list of unmatched track indices.
///     * A list of unmatched detection indices.
///
pub fn matching_cascade<D: Fn(&[Track], &[Detection], &[usize], &[usize]) -> Array2<f32>>(
    distance_metric: &D, 
    max_distance: f32, 
    cascade_depth: i32,
    tracks: &[Track], 
    detections: &[Detection], 
    track_indices: Option<Vec<usize>>, 
    detection_indices: Option<Vec<usize>>) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>)
{
    let track_indices = track_indices.unwrap_or_else(||(0..tracks.len()).collect());
    let detection_indices = detection_indices.unwrap_or_else(||(0..detections.len()).collect());
    let mut unmatched_detections = detection_indices.clone();
    let mut matches = vec![];

    for level in 0..cascade_depth {
        if unmatched_detections.is_empty() { // No detections left
            break;
        }

        let track_indices_l: Vec<_> = track_indices
            .iter()
            .copied()
            .filter(|&idx|tracks[idx].time_since_update == 1 + level)
            .collect();

        if track_indices_l.is_empty() {  // Nothing to match at this level
            continue;
        }

        let (mut matches_l, _, unmatched_detections_new) = min_cost_matching(
            distance_metric, max_distance, tracks, detections, 
            Some(track_indices_l), Some(unmatched_detections.clone()));
        
        unmatched_detections = unmatched_detections_new;

        matches.append(&mut matches_l);
    }

    let track_indices_set: HashSet<_> = track_indices.into_iter().collect();
    let matches_track_indices_set: HashSet<_> = matches.iter().map(|&(k, _)|k).collect();

    let unmatched_tracks: Vec<usize> = track_indices_set.difference(&matches_track_indices_set)
        .copied()
        .collect();

    (matches, unmatched_tracks, unmatched_detections)
}

/// Invalidate infeasible entries in cost matrix based on the state distributions obtained by Kalman filtering.
/// 
/// Parameters
/// ----------
/// kf : The Kalman filter.
/// cost_matrix : ndarray
///     The NxM dimensional cost matrix, where N is the number of track indices
///     and M is the number of detection indices, such that entry (i, j) is the
///     association cost between `tracks[track_indices[i]]` and
///     `detections[detection_indices[j]]`.
/// tracks : List[track.Track]
///     A list of predicted tracks at the current time step.
/// detections : List[detection.Detection]
///     A list of detections at the current time step.
/// track_indices : List[int]
///     List of track indices that maps rows in `cost_matrix` to tracks in
///     `tracks` (see description above).
/// detection_indices : List[int]
///     List of detection indices that maps columns in `cost_matrix` to
///     detections in `detections` (see description above).
/// gated_cost : Optional[float]
///     Entries in the cost matrix corresponding to infeasible associations are
///     set this value. Defaults to a very large value.
/// only_position : Optional[bool]
///     If True, only the x, y position of the state distribution is considered
///     during gating. Defaults to False.
/// 
/// Returns
/// -------
/// ndarray
///     Returns the modified cost matrix.
/// 
pub fn gate_cost_matrix(
    kf: &KalmanFilter, 
    mut cost_matrix: ArrayViewMut2<'_, f32>, 
    tracks: &[Track], 
    detections: &[Detection], 
    track_indices: &[usize], 
    detection_indices: &[usize],
    gated_cost: Option<f32>, 
    only_position: Option<bool>)
{
    let gated_cost = gated_cost.unwrap_or(INFTY_COST);
    let only_position = only_position.unwrap_or(false);
    let gating_dim = if only_position {1} else {3}; // indexes for 2 and 4 dims respectivly
    let gating_threshold = crate::sort::kalman_filter::CHI_2_INV_95[gating_dim];

    let mut measurements: Array2<f32> = unsafe { Array2::uninitialized((detection_indices.len(), 4)) };

    for (mut row, &idx) in measurements.axis_iter_mut(Axis(0)).zip(detection_indices.iter()) {
        let bbox = &detections[idx].bbox.as_xyah();

        row.assign(&bbox.as_view());
    }

    for (row, &track_idx) in track_indices.iter().enumerate() {
        let track = &tracks[track_idx];
        let gating_distance = kf.gating_distance(
            track.mean(), track.covariance(), measurements.view(), only_position);

        let mut axis = cost_matrix
            .index_axis_mut(Axis(0), row);
        
        for (idx, val) in axis.indexed_iter_mut() {
            if idx >= gating_distance.len() {
                break;
            }

            if gating_distance[idx] > gating_threshold {
                *val = gated_cost;
            }
        }
    }
}