use ndarray::prelude::*;
use crate::sort::{BBox, Ltwh};

/// 
/// This class represents a bounding box detection in a single image.
/// Parameters
/// 
/// ltwh : BBox in format `(x, y, w, h)`.
/// confidence : f32 - Detector confidence score.
/// feature : Vec<Array1<f32>> A feature vector that describes the object contained in this image.
///
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox<Ltwh>,
    pub confidence: f32, 
    pub feature: Option<Array1<f32>>
}