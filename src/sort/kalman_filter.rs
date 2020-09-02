use crate::sort::{BBox, Xyah};
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::triangular::*;

///
/// Table for the 0.95 quantile of the chi-square distribution with N degrees of
/// freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
/// function and used as Mahalanobis gating threshold.
///
pub const CHI_2_INV_95: [f32; 9] = [
    3.8415, // 1
    5.9915, // 2
    7.8147, // 3
    9.4877, // 4
    11.070, // 5
    12.592, // 6
    14.067, // 7
    15.507, // 8
    16.919, // 9
];

/// A simple Kalman filter for tracking bounding boxes in image space.
///
///     The 8-dimensional state space
///
///         x, y, a, h, vx, vy, va, vh
///
///     contains the bounding box center position (x, y), aspect ratio a, height h,
///     and their respective velocities.
///
///     Object motion follows a constant velocity model. The bounding box location
///     (x, y, a, h) is taken as direct observation of the state space (linear
///     observation model).
///
///     update_mat: Array2<f32> of shape (4, 8)
///
#[derive(Clone)]
pub struct KalmanFilter {
    ndim: usize,
    dt: f32,
    motion_mat: Array2<f32>,
    update_mat: Array2<f32>,
    std_weight_position: f32,
    std_weight_velocity: f32,
}

impl Default for KalmanFilter {
    fn default() -> Self {
        let (ndim, dt) = (4, 1.);

        // Create Kalman filter model matrices.
        let mut motion_mat = Array2::eye(2 * ndim);

        for i in 0..ndim {
            motion_mat[(i, ndim + i)] = dt;
        }

        let mut update_mat = Array2::zeros((ndim, 2 * ndim));

        for i in 0..ndim {
            update_mat[(i, i)] = 1.0;
        }

        // Motion and observation uncertainty are chosen relative to the current
        // state estimate. These weights control the amount of uncertainty in
        // the model. This is a bit hacky.
        let std_weight_position = 1.0 / 20.0;
        let std_weight_velocity = 1.0 / 160.0;

        Self {
            ndim,
            dt,
            motion_mat,
            update_mat,
            std_weight_position,
            std_weight_velocity,
        }
    }
}

impl KalmanFilter {
    ///
    /// Create track from unassociated measurement.
    ///
    ///     Parameters
    ///     ----------
    ///     measurement : ndarray
    ///         Bounding box coordinates (x, y, a, h) with center position (x, y),
    ///         aspect ratio a, and height h.
    ///
    ///     Returns
    ///     -------
    ///     (ndarray, ndarray)
    ///         Returns the mean vector (8 dimensional) and covariance matrix (8x8
    ///         dimensional) of the new track. Unobserved velocities are initialized
    ///         to 0 mean.
    ///
    pub fn initiate(&self, measurement: BBox<Xyah>) -> (Array1<f32>, Array2<f32>) {
        let mut mean = Array1::zeros((8,));
        mean.slice_mut(s![..4]).assign(&measurement.as_view());

        let std = arr1(&[
            2.0 * self.std_weight_position * measurement.height(),
            2.0 * self.std_weight_position * measurement.height(),
            1.0e-2,
            2.0 * self.std_weight_position * measurement.height(),
            10.0 * self.std_weight_velocity * measurement.height(),
            10.0 * self.std_weight_velocity * measurement.height(),
            1.0e-5,
            10.0 * self.std_weight_velocity * measurement.height(),
        ]);

        let covariance = Array2::from_diag(&(&std * &std));

        (mean, covariance)
    }

    /// Run Kalman filter prediction step.
    ///
    ///     Parameters
    ///     ----------
    ///     mean : ndarray
    ///         The 8 dimensional mean vector of the object state at the previous
    ///         time step.
    ///     covariance : ndarray
    ///         The 8x8 dimensional covariance matrix of the object state at the
    ///         previous time step.
    ///
    ///     Returns
    ///     -------
    ///     (ndarray, ndarray)
    ///         Returns the mean vector and covariance matrix of the predicted
    ///         state. Unobserved velocities are initialized to 0 mean.
    ///
    pub fn predict(&self, mean: ArrayView1<'_, f32>, covariance: ArrayView2<'_, f32>) -> (Array1<f32>, Array2<f32>) {
        let std = arr1(&[
            // posititon
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3],

            // velocity
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3],
        ]);

        let motion_cov = Array2::from_diag(&(&std * &std));
        let mean = self.motion_mat.dot(&mean);

        let covariance = self.motion_mat.dot(&covariance).dot(&self.motion_mat.t());

        (mean, covariance + motion_cov)
    }

    // Project state distribution to measurement space.
    //
    //     Parameters
    //     ----------
    //     mean : ndarray
    //         The state's mean vector (8 dimensional array).
    //     covariance : ndarray
    //         The state's covariance matrix (8x8 dimensional).
    //
    //     Returns
    //     -------
    //     (ndarray, ndarray)
    //         Returns the projected mean and covariance matrix of the given state
    //         estimate.
    //
    fn project(&self, mean: ArrayView1<'_, f32>, covariance: ArrayView2<'_, f32>) -> (Array1<f32>, Array2<f32>) {
        let std = arr1(&[
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3],
        ]);

        let innovation_cov = Array2::from_diag(&(&std * &std));
        let mean = self.update_mat.dot(&mean);
        let covariance = self.update_mat.dot(&covariance).dot(&self.update_mat.t());

        (mean, covariance + innovation_cov)
    }

    /// Run Kalman filter correction step.
    ///
    ///     Parameters
    ///     ----------
    ///     mean : ndarray
    ///         The predicted state's mean vector (8 dimensional).
    ///     covariance : ndarray
    ///         The state's covariance matrix (8x8 dimensional).
    ///     measurement : ndarray
    ///         The 4 dimensional measurement vector (x, y, a, h), where (x, y)
    ///         is the center position, a the aspect ratio, and h the height of the
    ///         bounding box.
    ///
    ///     Returns
    ///     -------
    ///     (ndarray, ndarray)
    ///         Returns the measurement-corrected state distribution.
    ///
    pub fn update(
        &self,
        mean: ArrayView1<'_, f32>,
        covariance: ArrayView2<'_, f32>,
        measurement: ArrayView1<'_, f32>,
    ) -> (Array1<f32>, Array2<f32>) {
        let (projected_mean, projected_cov) = self.project(mean, covariance);

        // chol shape (4, 4)
        let chol = projected_cov.factorizec(UPLO::Lower).unwrap();

        //       (8, 4)
        let mut kalman_gain = covariance.dot(&self.update_mat.t());

        for mut axis in kalman_gain.axis_iter_mut(Axis(0)) {
            // axis shape (4, )
            chol.solvec_inplace(&mut axis).unwrap();
        }

        let innovation = &measurement - &projected_mean;
        let new_mean = &mean + &innovation.dot(&kalman_gain.t());
        let new_covariance = &covariance - &kalman_gain.dot(&projected_cov).dot(&kalman_gain.t());

        (new_mean, new_covariance)
    }

    /// Compute gating distance between state distribution and measurements.
    ///
    ///     A suitable distance threshold can be obtained from `chi2inv95`. If
    ///     `only_position` is False, the chi-square distribution has 4 degrees of
    ///     freedom, otherwise 2.
    ///
    ///     Parameters
    ///     ----------
    ///     mean : ndarray
    ///         Mean vector over the state distribution (8 dimensional).
    ///     covariance : ndarray
    ///         Covariance of the state distribution (8x8 dimensional).
    ///     measurements : ndarray
    ///         An Nx4 dimensional matrix of N measurements, each in
    ///         format (x, y, a, h) where (x, y) is the bounding box center
    ///         position, a the aspect ratio, and h the height.
    ///     only_position : Optional[bool]
    ///         If True, distance computation is done with respect to the bounding
    ///         box center position only.
    ///
    ///     Returns
    ///     -------
    ///     ndarray
    ///         Returns an array of length N, where the i-th element contains the
    ///         squared Mahalanobis distance between (mean, covariance) and
    ///         `measurements[i]`.
    ///
    pub fn gating_distance(
        &self,
        mean: ArrayView1<'_, f32>,
        covariance: ArrayView2<'_, f32>,
        measurements: ArrayView2<'_, f32>,
        only_position: bool,
    ) -> Array1<f32> {
        let (mean, covariance) = self.project(mean, covariance);

        let (mean, covariance, measurements) = if only_position {
            (
                mean.slice(s!(..2)),
                covariance.slice(s!(..2, ..2)),
                measurements.slice(s!(.., ..2)),
            )
        } else {
            (mean.view(), covariance.view(), measurements.view())
        };

        let d = &measurements - &mean;

        let cholesky_lower = covariance.cholesky(UPLO::Lower).unwrap();
        let z = cholesky_lower
            .solve_triangular_into(UPLO::Lower, Diag::NonUnit, d.reversed_axes())
            .unwrap();

        (&z * &z).sum_axis(Axis(0))
    }
}

#[test]
fn test_kalman() {
    let mut kl = KalmanFilter::default();
    let (m, c) = kl.initiate(BBox::xyah(128.0, 128.0, 0.5, 64.0));

    // let target_mean = &[128. , 128. ,   0.5,  64. ,   0. ,   0. ,   0. ,   0. ]), array([[67.2   ,  0.    ,  0.    ,  0.    , 16.    ,  0.    ,  0.    ,
    //     0.    ],
    //   [ 0.    , 67.2   ,  0.    ,  0.    ,  0.    , 16.    ,  0.    ,
    //     0.    ],
    //   [ 0.    ,  0.    ,  0.0002,  0.    ,  0.    ,  0.    ,  0.    ,
    //     0.    ],
    //   [ 0.    ,  0.    ,  0.    , 67.2   ,  0.    ,  0.    ,  0.    ,
    //    16.    ],
    //   [16.    ,  0.    ,  0.    ,  0.    , 16.16  ,  0.    ,  0.    ,
    //     0.    ],
    //   [ 0.    , 16.    ,  0.    ,  0.    ,  0.    , 16.16  ,  0.    ,
    //     0.    ],
    //   [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,
    //     0.    ],
    //   [ 0.    ,  0.    ,  0.    , 16.    ,  0.    ,  0.    ,  0.    ,
    //    16.16  ]]))

    // println!("{:?}", kl.predict(m.view(),c.view()))

    let (m, c) = kl.update(m.view(),c.view(),aview1(&[192.0, 192.0, 0.5, 68.0]));

    println!("{:?}", kl.gating_distance(m.view(), c.view(), aview2(&[[256.0, 256.0, 0.5, 80.0]]), false));

}
