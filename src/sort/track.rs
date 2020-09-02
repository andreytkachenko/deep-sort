use ndarray::prelude::*;
use crate::sort::{Detection, KalmanFilter, BBox, Xyah};

///
///   Enumeration type for the single target track state. Newly created tracks are
///   classified as `tentative` until enough evidence has been collected. Then,
///   the track state is changed to `confirmed`. Tracks that are no longer alive
///   are classified as `deleted` to mark them for removal from the set of active
///   tracks.
///
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Deleted,
}

///
///     A single target track with state space `(x, y, a, h)` and associated
///     velocities, where `(x, y)` is the center of the bounding box, `a` is the
///     aspect ratio and `h` is the height.
///
///     Parameters
///     ----------
///     mean : ndarray
///         Mean vector of the initial state distribution.
///     covariance : ndarray
///         Covariance matrix of the initial state distribution.
///     track_id : int
///         A unique track identifier.
///     n_init : int
///         Number of consecutive detections before the track is confirmed. The
///         track state is set to `Deleted` if a miss occurs within the first
///         `n_init` frames.
///     max_age : int
///         The maximum number of consecutive misses before the track state is
///         set to `Deleted`.
///     feature : Optional[ndarray]
///         Feature vector of the detection this track originates from. If not None,
///         this feature is added to the `features` cache.
///
///     Attributes
///     ----------
///     mean : ndarray
///         Mean vector of the initial state distribution.
///     covariance : ndarray
///         Covariance matrix of the initial state distribution.
///     track_id : int
///         A unique track identifier.
///     hits : int
///         Total number of measurement updates.
///     age : int
///         Total number of frames since first occurance.
///     time_since_update : int
///         Total number of frames since last measurement update.
///     state : TrackState
///         The current track state.
///     features : List[ndarray]
///         A cache of features. On each measurement update, the associated feature
///         vector is added to this list.
///
#[derive(Clone)]
pub struct Track {
    pub track_id: i32,
    pub time_since_update: i32,
    pub features: Vec<Array1<f32>>,

    covariance: Array2<f32>,
    mean: Array1<f32>,
    hits: i32,
    age: i32,
    state: TrackState,
    n_init: i32,
    max_age: i32,
}

impl Track {
    pub fn new(mean: Array1<f32>, covariance: Array2<f32>, track_id: i32, n_init: i32, max_age: i32, feature: Option<Array1<f32>>) -> Self {
        Self {
            track_id,
            mean,
            covariance,
            hits: 1,
            age: 1,
            time_since_update: 0,
            state: TrackState::Tentative,
            features: feature.into_iter().collect(),
            n_init,
            max_age,
        }
    }

    /// Get current position in bounding box format `(top left x, top left y, width, height)`.
    ///
    ///     Returns
    ///     -------
    ///     ndarray
    ///         The bounding box.
    ///
    #[inline]
    pub fn bbox(&self) -> BBox<Xyah> {
        BBox::xyah(
            self.mean[0],
            self.mean[1],
            self.mean[2],
            self.mean[3],
        )
    }

    #[inline]
    pub fn mean(&self) -> ArrayView1<'_, f32> {
        self.mean.view()
    }

    #[inline]
    pub fn covariance(&self) -> ArrayView2<'_, f32> {
        self.covariance.view()
    }

    /// Propagate the state distribution to the current time step using a
    ///     Kalman filter prediction step.
    ///
    ///     Parameters
    ///     ----------
    ///     kf : kalman_filter.KalmanFilter
    ///         The Kalman filter.
    ///
    pub fn predict(&mut self, kf: &mut KalmanFilter) {
        let (mean, covariance) = kf.predict(self.mean.view(), self.covariance.view());
        self.mean = mean;
        self.covariance = covariance;
        self.age += 1;
        self.time_since_update += 1;
    }

    /// Perform Kalman filter measurement update step and update the feature cache.
    ///
    ///     Parameters
    ///     ----------
    ///     kf : kalman_filter.KalmanFilter
    ///         The Kalman filter.
    ///     detection : Detection
    ///         The associated detection.
    ///
    pub fn update(&mut self, kf: &mut KalmanFilter, detection: &Detection) {
        let (mean, covariance) = kf.update(
            self.mean.view(),
            self.covariance.view(),
            detection.bbox.as_xyah().as_view()
        );

        self.mean = mean;
        self.covariance = covariance;
        self.features.extend(detection.feature.clone().into_iter());

        self.hits += 1;
        self.time_since_update = 0;

        if self.state == TrackState::Tentative && self.hits >= self.n_init {
            self.state = TrackState::Confirmed;
        }
    }

    ///
    /// Mark this track as missed (no association at the current time step).
    ///
    #[inline]
    pub fn mark_missed(&mut self) {
        if self.state == TrackState::Tentative {
            self.state = TrackState::Deleted;
        } else if self.time_since_update > self.max_age {
            self.state = TrackState::Deleted;
        }
    }
    ///
    /// Returns True if this track is tentative (unconfirmed).
    ///
    #[inline]
    pub fn is_tentative(&self) -> bool {
        self.state == TrackState::Tentative
    }

    ///
    /// Returns True if this track is confirmed.
    ///
    #[inline]
    pub fn is_confirmed(&self) -> bool {
        self.state == TrackState::Confirmed
    }

    ///
    /// Returns True if this track is dead and should be deleted.
    ///
    #[inline]
    pub fn is_deleted(&self) -> bool {
        self.state == TrackState::Deleted
    }
}
