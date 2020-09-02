use std::collections::{VecDeque, HashMap};
use ndarray::prelude::*;
use crate::sort::DistanceMetric;
use crate::sort::linear_assignment::INFTY_COST;

// Compute pair-wise squared distance between points in `a` and `b`.
//
//     Parameters
//     ----------
//     a : array_like
//         An NxM matrix of N samples of dimensionality M.
//     b : array_like
//         An LxM matrix of L samples of dimensionality M.
//
//     Returns
//     -------
//     ndarray
//         Returns a matrix of size len(a), len(b) such that eleement (i, j)
//         contains the squared distance between `a[i]` and `b[j]`.
//
fn pdist(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>) -> Array2<f32> {
    if a.is_empty() || b.is_empty() {
        return Array2::zeros((a.len(), b.len()));
    }

    let (a2, b2) = (
        (&a * &a).sum_axis(Axis(1)).insert_axis(Axis(1)),
        (&b * &b).sum_axis(Axis(1)).insert_axis(Axis(0))
    );

    let mut r2 = -2.0 * a.dot(&b.t()) + a2 + b2;

    r2.mapv_inplace(|x| if x < 0.0 {
        0.0
    } else {
        x
    });

    r2
}

/// Compute pair-wise cosine distance between points in `a` and `b`.
///
///     Parameters
///     ----------
///     a : array_like
///         An NxM matrix of N samples of dimensionality M.
///     b : array_like
///         An LxM matrix of L samples of dimensionality M.
///     data_is_normalized : Optional[bool]
///         If True, assumes rows in a and b are unit length vectors.
///         Otherwise, a and b are explicitly normalized to lenght 1.
///
///     Returns
///     -------
///     ndarray
///         Returns a matrix of size len(a), len(b) such that eleement (i, j)
///         contains the squared distance between `a[i]` and `b[j]`.
///
fn cosine_distance(a: ArrayView2<'_, f32>, b: ArrayView2<'_, f32>, data_is_normalized: bool) -> Array2<f32> {
    if data_is_normalized {
        -a.dot(&b.t()) + 1.0
    } else {
        let length_a = a.map_axis(Axis(1), |x|x.fold(0.0, |a, x|a + x*x).sqrt());
        let length_b = b.map_axis(Axis(1), |x|x.fold(0.0, |a, x|a + x*x).sqrt());

        let a = &a / &length_a.insert_axis(Axis(1));
        let b = &b / &length_b.insert_axis(Axis(1));

        -a.dot(&b.t()) + 1.0
    }
}


#[test]
fn cosine_distance_test() {
    println!("{:?}", cosine_distance(aview2(&[[1.0f32,2.,3.,4.,5.,6.]]), aview2(&[[5.,6.,7.,8.,9.,10.]]), false));
}

/// Helper function for nearest neighbor distance metric (Euclidean).
///
///     Parameters
///     ----------
///     x : ndarray
///         A matrix of N row-vectors (sample points).
///     y : ndarray
///         A matrix of M row-vectors (query points).
///
///     Returns
///     -------
///     ndarray
///         A vector of length M that contains for each entry in `y` the
///         smallest Euclidean distance to a sample in `x`.
///
fn nn_euclidean_distance(x: ArrayView2<'_, f32>, y: ArrayView2<'_, f32>) -> Array1<f32> {
    let distances = pdist(x, y);

    distances.map_axis(Axis(0), |view|view.fold(f32::MAX, |a, &x| if x < a { x } else { a }))
        .mapv_into(|x|x.max(0.0))
}

/// Helper function for nearest neighbor distance metric (cosine).
///
///     Parameters
///     ----------
///     x : ndarray
///         A matrix of N row-vectors (sample points).
///     y : ndarray
///         A matrix of M row-vectors (query points).
///
///     Returns
///     -------
///     ndarray
///         A vector of length M that contains for each entry in `y` the
///         smallest cosine distance to a sample in `x`.
///
fn nn_cosine_distance(x: ArrayView2<'_, f32>, y: ArrayView2<'_, f32>) -> Array1<f32> {
    let distances = cosine_distance(x, y, false);

    distances.map_axis(Axis(0), |view|view.fold(f32::MAX, |a, &x| if x < a { x } else { a }))
}

#[derive(Clone)]
pub enum NearestNeighborMetricKind {
    EuclideanDistance,
    CosineDistance,
}

///
/// A nearest neighbor distance metric that, for each target, returns
/// the closest distance to any sample that has been observed so far.
///
///     Parameters
///     ----------
///     metric : str
///         Either "euclidean" or "cosine".
///     matching_threshold: float
///         The matching threshold. Samples with larger distance are considered an
///         invalid match.
///     budget : Optional[int]
///         If not None, fix samples per class to at most this number. Removes
///         the oldest samples when the budget is reached.
///
///     Attributes
///     ----------
///     samples : Dict[int -> List[ndarray]]
///         A dictionary that maps from target identities to the list of samples
///         that have been observed so far.
///
#[derive(Clone)]
pub struct NearestNeighborDistanceMetric {
    metric_kind: NearestNeighborMetricKind,
    matching_threshold: f32,
    budget: Option<usize>,
    samples: HashMap<i32, VecDeque<Array1<f32>>>,
}

impl NearestNeighborDistanceMetric {
    pub fn new(metric_kind: NearestNeighborMetricKind, matching_threshold: f32, budget: Option<usize>) -> Self {
        Self {
            metric_kind,
            matching_threshold,
            budget,
            samples: Default::default(),
        }
    }
}

impl DistanceMetric for NearestNeighborDistanceMetric {
    #[inline]
    fn matching_threshold(&self) -> f32 {
        self.matching_threshold
    }

    /// Update the distance metric with new data.
    ///
    ///     Parameters
    ///     ----------
    ///     features : ndarray
    ///         An NxM matrix of N features of dimensionality M.
    ///     targets : ndarray
    ///         An integer array of associated target identities.
    ///     active_targets : List[int]
    ///         A list of targets that are currently present in the scene.
    ///
    fn partial_fit(&mut self, features: Vec<Array1<f32>>, targets: Vec<i32>, active_targets: Vec<i32>) {
        for (feature, target) in features.into_iter().zip(targets.into_iter()) {
            let deque = self.samples
                .entry(target)
                .or_insert_with(VecDeque::new);

            deque.push_front(feature);

            if let Some(budget) = self.budget {
                deque.truncate(budget);
            }
        }

        let new_samples = active_targets
            .into_iter()
            .filter_map(|k| Some((k, self.samples.remove(&k)?)))
            .collect();

        self.samples = new_samples;
    }

    /// Compute distance between features and targets.
    ///
    ///     Parameters
    ///     ----------
    ///     features : ndarray
    ///         An NxM matrix of N features of dimensionality M.
    ///     targets : List[int]
    ///         A list of targets to match the given `features` against.
    ///
    ///     Returns
    ///     -------
    ///     ndarray
    ///         Returns a cost matrix of shape len(targets), len(features), where
    ///         element (i, j) contains the closest squared distance between
    ///         `targets[i]` and `features[j]`.
    ///
    fn distance(&self, features: ArrayView2<'_, f32>, targets: Vec<i32>) -> Array2<f32> {
        let ntargets = targets.len();
        let nfeatures = features.nrows();
        let n = nfeatures.max(ntargets);

        let mut cost_matrix = Array2::from_elem((n, n), INFTY_COST);

        for (i, target) in targets.into_iter().enumerate() {
            let sample_features_deq = &self.samples[&target];

            let mut sample_features = unsafe { Array::uninitialized((sample_features_deq.len(), features.ncols())) };
            sample_features_deq
                .iter()
                .enumerate()
                .for_each(|(idx, arr)| sample_features.index_axis_mut(Axis(0), idx).assign(&arr));

            cost_matrix
                .slice_mut(s![i, ..nfeatures])
                .assign(&match self.metric_kind {
                    NearestNeighborMetricKind::EuclideanDistance => nn_euclidean_distance(sample_features.view(), features.view()),
                    NearestNeighborMetricKind::CosineDistance => nn_cosine_distance(sample_features.view(), features.view()),
                });
        }

        cost_matrix
    }
}
