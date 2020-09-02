pub mod deep;
pub mod sort;
pub mod error;


pub use sort::Track;
use deep::ImageEncoder;
use sort::{Tracker, NearestNeighborMetricKind, NearestNeighborDistanceMetric};
use ndarray::prelude::*;
use opencv::core::{self, Mat, Rect};
use opencv::core::MatTrait;

use opencv::dnn;
use error::Error;
use grant_object_detector as detector;

use std::collections::HashMap;

fn get_detection_cv_rect(det: &detector::Detection, frame_width: i32, frame_height: i32) -> Rect {
    let left = det.xmin.max(0).min(frame_width);
    let top = det.ymin.max(0).min(frame_height);

    let mut width = (det.xmax - det.xmin).max(0);
    let mut height = (det.ymax - det.ymin).max(0);

    if left + width > frame_width {
        width = frame_width - left;
    }

    if top + height > frame_height {
        height = frame_height - top;
    }

    Rect::new(
        left,
        top,
        width,
        height,
    )
}

pub struct DeepSortConfig {
    pub reid_model_path: String,
    pub max_cosine_distance: f32,
    pub nn_budget: Option<usize>,
    pub max_age: i32,
    pub max_iou_distance: f32,
    pub n_init: i32,
}

impl DeepSortConfig {
    pub fn new(reid_model_path: String) -> Self {
        Self {
            reid_model_path,
            max_cosine_distance: 0.2,
            nn_budget: Some(100),
            max_age: 70,
            max_iou_distance: 0.7,
            n_init: 3
        }
    }
}

pub struct DeepSort {
    device: onnx_model::OnnxInferenceDevice,
    encoder: ImageEncoder,
    sample_tracker: Tracker<NearestNeighborDistanceMetric>,
    trackers: HashMap<String, Tracker<NearestNeighborDistanceMetric>>,
}

impl DeepSort {
    pub fn new(config: DeepSortConfig) -> Result<Self, Error> {
        let metric = NearestNeighborDistanceMetric::new(
            NearestNeighborMetricKind::CosineDistance,
            config.max_cosine_distance,
            config.nn_budget
        );

        let device = onnx_model::get_cuda_if_available();

        Ok(Self {
            device,
            sample_tracker: Tracker::new(
                metric,
                config.max_iou_distance,
                config.max_age,
                config.n_init
            ),
            encoder: ImageEncoder::new(&config.reid_model_path, device)?,
            trackers: HashMap::<String, Tracker<NearestNeighborDistanceMetric>>::new(),
        })
    }

    #[inline]
    pub fn tracks(&self, src_url: String) -> &[sort::Track] {
        self.trackers.get(&src_url)
            .unwrap()
            .tracks()
    }

    pub fn update(&mut self, frames: &[Mat], dets: &[Vec<detector::Detection>], src_url: String) -> Result<(), Error> {
        let total_count = dets.iter().map(|i|i.len()).sum::<usize>();

        let tracker = self.trackers.entry(src_url.clone()).or_insert(self.sample_tracker.clone());

        if total_count == 0 {
            return Ok(());
        }

        let input_shape = self.encoder.input_shape();
        let frag_channs = input_shape[0] as usize;
        let frag_height = input_shape[1] as usize;
        let frag_width = input_shape[2] as usize;

        let mut idets = unsafe { Array4::uninitialized([total_count, frag_channs, frag_height, frag_width]) };
        let mut index = 0usize;
        for (frame, dets) in frames.iter().zip(dets.iter()) {
            for det in dets {
                let rect = get_detection_cv_rect(&det, frame.cols(), frame.rows());
                let roi = Mat::roi(&frame, rect).unwrap();
                let blob = dnn::blob_from_image(
                    &roi,
                    1.0 / 255.0,
                    core::Size::new(frag_width as i32, frag_height as i32),
                    core::Scalar::new(0., 0., 0., 0.),
                    true,
                    false,
                    core::CV_32F).unwrap();

                let core = blob.try_into_typed::<f32>().unwrap();
                let data: &[f32] = core.data_typed().unwrap();
                let a = aview1(data).into_shape((frag_channs, frag_height, frag_width)).unwrap();
                idets.index_axis_mut(Axis(0), index).assign(&a);
                index += 1;
            }
        }

        let features = self.encoder.encode_batch(idets.view())?;
        // let features = Array2::zeros((dets.len(), 512));
        let mut detections = vec![];

        for dets in dets.iter() {
            detections.clear();
            for (feature, det) in features.axis_iter(Axis(0)).zip(dets.iter()) {
                let x = sort::Detection {
                    bbox: sort::BBox::ltwh(
                        det.xmin as _,
                        det.ymin as _,
                        (det.xmax - det.xmin) as _,
                        (det.ymax - det.ymin) as _,
                    ),
                    confidence: det.confidence,
                    feature: Some(feature.into_owned())
                };

                detections.push(x);
            }

            tracker.predict();
            tracker.update(detections.as_slice());
        }

        Ok(())
    }
}
