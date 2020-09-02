pub mod detection;
pub mod iou_matching;
pub mod kalman_filter;
pub mod nn_matching;
pub mod linear_assignment;
pub mod tracker;
pub mod track;

pub use detection::Detection;
pub use kalman_filter::KalmanFilter;
pub use tracker::Tracker;
pub use track::{Track, TrackState};
pub use nn_matching::*;

use core::marker::PhantomData;
use ndarray::prelude::*;

pub trait BBoxFormat: std::fmt::Debug {}

#[derive(Debug, Copy, Clone)]
pub struct Ltwh;
impl BBoxFormat for Ltwh {}

#[derive(Debug, Copy, Clone)]
pub struct Xyah;
impl BBoxFormat for Xyah {}

#[derive(Debug, Copy, Clone)]
pub struct Ltrb;
impl BBoxFormat for Ltrb {}


#[derive(Debug, Clone)]
pub struct BBox<F: BBoxFormat>([f32; 4], PhantomData<F>);
impl<F: BBoxFormat> BBox<F> {
    #[inline]
    pub fn as_view(&self) -> ArrayView1<'_, f32> {
        aview1(&self.0)
    }

    #[inline]
    pub fn xyah(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox(
            [x1, x2, x3, x4],
            Default::default(),
        )
    }
}

impl BBox<Ltwh> {
    #[inline(always)]
    pub fn left(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn top(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn width(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.0[3]
    }

    #[inline]
    pub fn as_xyah(&self) -> BBox<Xyah> {
        self.into()
    }

    #[inline]
    pub fn as_ltrb(&self) -> BBox<Ltrb> {
        self.into()
    }

    #[inline]
    pub fn ltwh(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox(
            [x1, x2, x3, x4],
            Default::default(),
        )
    }
}

impl BBox<Ltrb> {
    #[inline]
    pub fn ltrb(x1: f32, x2: f32, x3: f32, x4: f32) -> Self {
        BBox(
            [x1, x2, x3, x4],
            Default::default(),
        )
    }

    #[inline]
    pub fn as_ltwh(&self) -> BBox<Ltwh> {
        self.into()
    }

    #[inline(always)]
    pub fn left(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn top(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn right(&self) -> f32 {
        self.0[2]
    }

    #[inline(always)]
    pub fn bottom(&self) -> f32 {
        self.0[3]
    }
}

impl BBox<Xyah> {
    #[inline(always)]
    pub fn as_ltrb(&self) -> BBox<Ltrb> {
        self.into()
    }

    #[inline(always)]
    pub fn as_ltwh(&self) -> BBox<Ltwh> {
        self.into()
    }

    #[inline(always)]
    pub fn cx(&self) -> f32 {
        self.0[0]
    }

    #[inline(always)]
    pub fn cy(&self) -> f32 {
        self.0[1]
    }

    #[inline(always)]
    pub fn height(&self) -> f32 {
        self.0[3]
    }
}

impl <'a> From<&'a BBox<Ltwh>> for BBox<Xyah> {
    #[inline]
    fn from(v: &'a BBox<Ltwh>) -> Self {
        Self([
            v.0[0] + v.0[2] / 2.0, 
            v.0[1] + v.0[3] / 2.0, 
            v.0[2] / v.0[3], 
            v.0[3],
        ], Default::default())
    }
}

impl <'a> From<&'a BBox<Ltwh>> for BBox<Ltrb> {
    #[inline]
    fn from(v: &'a BBox<Ltwh>) -> Self {
        Self([
            v.0[0], 
            v.0[1], 
            v.0[2] + v.0[0], 
            v.0[3] + v.0[1],
        ], Default::default())
    }
}

impl <'a> From<&'a BBox<Ltrb>> for BBox<Ltwh> {
    #[inline]
    fn from(v: &'a BBox<Ltrb>) -> Self {
        Self([
            v.0[0], 
            v.0[1], 
            v.0[2] - v.0[0], 
            v.0[3] - v.0[1],
        ], Default::default())
    }
}

impl <'a> From<&'a BBox<Xyah>> for BBox<Ltwh> {
    #[inline]
    fn from(v: &'a BBox<Xyah>) -> Self {
        let height = v.0[3];
        let width = v.0[2] * height;

        Self([
            v.0[0] - width / 2.0, 
            v.0[1] - height / 2.0, 
            width, 
            height,
        ], Default::default())
    }
}

impl <'a> From<&'a BBox<Xyah>> for BBox<Ltrb> {
    #[inline]
    fn from(v: &'a BBox<Xyah>) -> Self {
        (&v.as_ltwh()).into()
    }
}

pub trait DistanceMetric {

    /// Getting the matching threshold
    ///
    fn matching_threshold(&self) -> f32;

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
    fn partial_fit(&mut self, features: Vec<Array1<f32>>,  targets: Vec<i32>, active_targets: Vec<i32>);

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
    fn distance(&self, features: ArrayView2<'_, f32>, targets: Vec<i32>) -> Array2<f32>;
}