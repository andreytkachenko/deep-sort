use opencv::{
    dnn,
    core::{self, Mat, Scalar, Vector},
    highgui,
    prelude::*,
    videoio,
};
use deep_sort::{
    deep::{ImageEncoder, ORT_ENV},
    sort,
};
use ndarray::prelude::*;
use onnxruntime::*;
use std::borrow::Borrow;

const SIZE: usize = 416;
const OUT_SIZE: usize = 10647; //6300; //16128 10647;

const CHANNELS: usize = 24;
const CONFIDENCE_THRESHOLD: f32 = 0.5;
const NMS_THRESHOLD: f32 = 0.4;
pub const NAMES: [&'static str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

pub enum Target {
    Cpu,
    Cuda,
    Tensorrt,
    Movidus
}

struct Inference {

}

impl Inference {
    pub fn new(model: &str, config: &str) -> opencv::Result<Self> {
        unimplemented!()
    }

    pub fn set_preferable_target(&mut self, target: Target) {

    }

    pub fn set_input(&mut self, name: &str, tensor: Tensor<f32>) {

    }

    pub fn get_output(&self, name: &str) {

    }

    pub fn forward(&mut self) {

    }
}

#[derive(Debug)]
pub enum Error {
    OpenCv(opencv::Error),
    OnnxRuntime(onnxruntime::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenCv(i) => writeln!(f, "Error: {}", i),
            Self::OnnxRuntime(i) => writeln!(f, "Error: {}", i),
        }
    }
}

impl std::error::Error for Error {}

impl From<opencv::Error> for Error {
    fn from(err: opencv::Error) -> Self {
        Self::OpenCv(err)
    }
}


impl From<onnxruntime::Error> for Error {
    fn from(err: onnxruntime::Error) -> Self {
        Self::OnnxRuntime(err)
    }
}


fn run() -> std::result::Result<(), Error> {
    let max_cosine_distance = 0.2;
    let nn_budget = 100;
    let max_age = 70;
    let max_iou_distance = 0.7;
    let n_init = 3;
    let kind = sort::NearestNeighborMetricKind::CosineDistance;
    let metric = sort::NearestNeighborDistanceMetric::new(kind, max_cosine_distance, Some(nn_budget));
    let mut tracker = sort::Tracker::new(metric, max_iou_distance, max_age, n_init);

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let mut cam = videoio::VideoCapture::from_file("./videoplayback.mp4", videoio::CAP_ANY)?;  // 0 is the default camera
    cam.set(videoio::CAP_PROP_POS_FRAMES, 150.0);

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut frame = core::Mat::default()?;
    let mut flag = -1i64;

    let (in_tx, in_rx): (std::sync::mpsc::Sender<(Tensor<f32>, Mat)>, _) = std::sync::mpsc::channel();
    let (out_tx, out_rx) = std::sync::mpsc::channel();
    let (bb_tx, bb_rx) = std::sync::mpsc::channel();

    std::thread::spawn(move || {
        let mut so = SessionOptions::new().unwrap();
        let ro = RunOptions::new();
    
        // so.set_execution_mode(ExecutionMode::Parallel).unwrap();
        // so.add_tensorrt(0);
        so.add_cuda(0);
        // so.add_cpu(true);
    
        let session = Session::new(&ORT_ENV, "/home/andrey/workspace/ssl/yolov4/yolov4_416.onnx", &so).unwrap();
        let mut out_vals = Tensor::<f32>::init(&[1, OUT_SIZE as _, 84], 0.0).unwrap();
        
        let input = std::ffi::CStr::from_bytes_with_nul(b"input\0").unwrap();
        let output = std::ffi::CStr::from_bytes_with_nul(b"output\0").unwrap();

        while let Ok((in_vals, frame)) = in_rx.recv() {
            let in_vals: Tensor<f32> = in_vals;
            session
                .run_mut(&ro, &[input], &[in_vals.as_ref()], &[output], &mut [out_vals.as_mut()])
                .expect("run");

            let xx: &[f32] = out_vals.borrow();

            let arr = Array3::from_shape_vec([1, OUT_SIZE, 84], xx.to_vec()).unwrap();

            out_tx.send((arr, frame)).unwrap();
        }
    });

    std::thread::spawn(move || {
        let mut encoder = ImageEncoder::new("/home/andrey/workspace/ssl/deep_sort_pytorch/deep_sort/deep/reid.onnx").unwrap();
    
        while let Ok((out_vals, frame)) = out_rx.recv() {
            let mut preds = vec![];
            let mut tracks = vec![];

            // let mut detections = vec![];
            // let out_shape = out_vals.shape();
            let bboxes = detect(out_vals.view()).unwrap();

            let nboxes = bboxes.len();
            // let mut in_vals = Array4::from_elem([nboxes, 3, 128, 64], 0.0);

            // for (index, bbox) in bboxes.iter().enumerate() {
            //     let rect = bbox.cv_rect(frame.cols(), frame.rows());
                
            //     let roi = Mat::roi(&frame, rect).unwrap();
            //     let blob = dnn::blob_from_image(
            //         &roi, 
            //         1.0 / 255.0, 
            //         core::Size::new(64, 128), 
            //         core::Scalar::new(0., 0., 0., 0.), 
            //         true, 
            //         false, 
            //         core::CV_32F)
            //         .unwrap();

            //     let core = blob.into_typed::<f32>().unwrap();
            //     let data: &[f32] = core.data_typed().unwrap();

            //     let a = aview1(data).into_shape((3, 128, 64)).unwrap();
            //     in_vals.index_axis_mut(Axis(0), index)
            //         .assign(&a)
            // }

            // let t = TensorView::new(in_vals.shape(), in_vals.as_slice().unwrap());
            // let code = encoder.encode_batch(t).unwrap();
            // let features = aview1(code.borrow()).into_shape((nboxes, 512)).unwrap();

            for (i, bbox) in bboxes.into_iter().enumerate() {
                let rect = bbox.cv_rect(frame.cols(), frame.rows());
                // let feature = features.index_axis(Axis(0), i);

                if bbox.class_index <= 8 {
                    // detections.push(sort::Detection {
                    //     bbox: sort::BBox::ltwh(
                    //         rect.x as f32, 
                    //         rect.y as f32, 
                    //         rect.width as f32, 
                    //         rect.height as f32
                    //     ),
                    //     confidence: bbox.confidence, 
                    //     feature: Some(feature.into_owned())
                    // });

                    preds.push(bbox);
                }
            }

            // tracker.predict();
            // tracker.update(detections.as_slice());

            // for t in tracker.tracks().iter().filter(|t| t.is_confirmed() && t.time_since_update <= 1) {
            //     tracks.push((t.bbox().as_ltwh(), t.track_id));
            //     // draw_track(&mut frame, t.bbox().as_ltwh(), t.track_id);
            // }

            bb_tx.send((preds, tracks)).unwrap();
        }
    });

    loop {
        let begin = std::time::Instant::now();
        cam.read(&mut frame)?;

        let fsize = frame.size()?;

        if fsize.width <= 0 {
            continue;
        }

        let frame_height = fsize.height;
        let frame_width = fsize.width;

        //  Create a 4D blob from a frame.
        let inp_width = SIZE as _;
        let inp_height = SIZE as _;
        let blob = dnn::blob_from_image(
            &frame, 
            1.0 / 255.0, 
            core::Size::new(inp_width, inp_height), 
            core::Scalar::new(0., 0., 0., 0.), 
            true, 
            false, 
            core::CV_32F)
            .unwrap();
        
        let core = blob.try_into_typed::<f32>()?;
        let data: &[f32] = core.data_typed()?;
        let in_vals = Tensor::new(&[1, 3, SIZE as _, SIZE as _], data.to_vec()).unwrap();

        // //  Run a model

        in_tx.send((in_vals, frame.clone()?)).unwrap();
        let (preds, tracks) = bb_rx.recv().unwrap();
        
        for p in preds {
            draw_pred(&mut frame, p)?;
        }

        for (t, i) in tracks {
            draw_track(&mut frame, t, i);
        }

        // let mut objects = vec![];
        
        // let mut detections = vec![];
        // // let out_shape = out_vals.dims();
        // let bboxes = detect(aview1(out_vals.borrow())
        //     .into_shape([out_shape[0] as usize, out_shape[1] as usize, out_shape[2]as usize]).unwrap())?;

        // for bbox in bboxes {
        //     let rect = bbox.cv_rect(frame.cols(), frame.rows());
            
        //     let roi = Mat::roi(&frame, rect)?;
        //     let blob = dnn::blob_from_image(
        //         &roi, 
        //         1.0 / 255.0, 
        //         core::Size::new(64, 128), 
        //         core::Scalar::new(0., 0., 0., 0.), 
        //         true, 
        //         false, 
        //         core::CV_32F)
        //         .unwrap();
            
        //     let code = encoder.encode_batch(&blob)?.get(0)?;
        //     let core = code.into_typed::<f32>()?;
        //     let feature = arr1(core.data_typed()?);

        //     if bbox.class_index <= 8 {
        //         detections.push(sort::Detection {
        //             bbox: sort::BBox::ltwh(
        //                 rect.x as f32, 
        //                 rect.y as f32, 
        //                 rect.width as f32, 
        //                 rect.height as f32
        //             ),
        //             confidence: bbox.confidence, 
        //             feature: Some(feature)
        //         });

        //         draw_pred(&mut frame, bbox)?;
        //     }
        // }

        // tracker.predict();
        // tracker.update(detections.as_slice());

        // for t in tracker.tracks().iter().filter(|t| t.is_confirmed() && t.time_since_update <= 1) {
        //     draw_track(&mut frame, t.bbox().as_ltwh(), t.track_id);
        // }

        let diff = std::time::Instant::now() - begin;
        let label = format!("{:?}", 1.0 / ((diff.as_millis() as f32) * 0.001));

        opencv::imgproc::put_text(
            &mut frame, 
            &label, 
            core::Point::new(30, 30), 
            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
            0.6, 
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            opencv::imgproc::LINE_8, 
            false
        )?;

        highgui::imshow(window, &mut frame)?;

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}


fn detect(pred_events: &ArrayView3<'_, f32>) -> opencv::Result<Vec<BBox>> {

    // The bounding boxes grouped by (maximum) class index.
    let mut bboxes: Vec<(core::Vector<core::Rect2d>, core::Vector<f32>, Vec<BBox>)> = (0 .. 80)
        .map(|_| (core::Vector::new(), core::Vector::new(), vec![]))
        .collect();

    for pred_event in pred_events {
        let fsize = pred_event.size()?;
        let npreds = pred_event.rows();
        let pred_size = pred_event.cols();
        let nclasses = (pred_size - 5) as usize;

        // Extract the bounding boxes for which confidence is above the threshold.
        for index in 0 .. npreds {
            let pred = pred_event.row(index)?.try_into_typed::<f32>()?;
            let detection = pred.data_typed()?;

            let (center_x, center_y, width, height, confidence) = match &detection[0 .. 5] {
                &[a,b,c,d,e] => (a,b,c,d,e),
                _ => unreachable!()
            };

            let classes = &detection[5..];

            if confidence > CONFIDENCE_THRESHOLD {
                let mut class_index = -1;
                let mut score = 0.0;

                for (idx, &val) in classes.iter().enumerate() {
                    if val > score {
                        class_index = idx as i32;
                        score = val;
                    }
                }

                if class_index > -1 && score > 0. {
                    let entry = &mut bboxes[class_index as usize];

                    entry.0.push(core::Rect2d::new(
                        (center_x - width / 2.) as f64,
                        (center_y - height / 2.) as f64,
                        width as f64,
                        height as f64,
                    ));
                    entry.1.push(score);
                    entry.2.push(BBox {
                        xmin: center_x - width / 2.,
                        ymin: center_y - height / 2.,
                        xmax: center_x + width / 2.,
                        ymax: center_y + height / 2.,
                        
                        confidence,
                        class_index: class_index as _,
                        class_confidence: score,
                    });
                }
            }
        }
    }

    let mut events = vec![];

    for (rects, scores, bboxes) in bboxes.iter_mut() {
        if bboxes.is_empty() {
            continue;
        }

        let mut indices = core::Vector::<i32>::new();
        dnn::nms_boxes_f64(
            &rects, 
            &scores, 
            CONFIDENCE_THRESHOLD, 
            NMS_THRESHOLD, 
            &mut indices, 
            1.0, 
            0
        )?;

        let mut indices = indices.to_vec();

        events.extend(bboxes.drain(..)
            .enumerate()
            .filter_map(|(idx, item)| if indices.contains(&(idx as i32)) {Some(item)} else {None}));
    }

    // Perform non-maximum suppression.
    // for (idx, (_, _, bboxes_for_class)) in bboxes.iter_mut().enumerate() {
    //     if bboxes_for_class.is_empty() {
    //         continue;
    //     }

    //     bboxes_for_class.sort_unstable_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
    //     let mut current_index = 0;

    //     for index in 0 .. bboxes_for_class.len() {
    //         let mut drop = false;
    //         for prev_index in 0..current_index {
    //             let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
    //             if iou > NMS_THRESHOLD {
    //                 drop = true;
    //                 break;
    //             }
    //         }

    //         if !drop {
    //             bboxes_for_class.swap(current_index, index);
    //             current_index += 1;
    //         }
    //     }

    //     bboxes_for_class.truncate(current_index);
    // }

    for (class_index, (_, _, bboxes_for_class)) in bboxes.into_iter().enumerate() {
        if bboxes_for_class.is_empty() {
            continue;
        }

        let clamp = |x| if x < 0.0 { 0.0 } else if x > 1.0 { 1.0 } else { x };

        for bbox in bboxes_for_class {
            events.push(bbox);
        }
    }

    Ok(events)
}

fn draw_pred(frame: &mut Mat, bbox: BBox) -> opencv::Result<()> {
    let rect = bbox.cv_rect(frame.cols(), frame.rows());

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame, 
        rect, 
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), 
        1, 
        opencv::imgproc::LINE_8, 
        0
    )?;

    // let label = format!("{} {:2}", NAMES[bbox.class_index], bbox.class_confidence);
    // let mut base_line = 0; 
    // let label_size = opencv::imgproc::get_text_size(&label, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.6, 1, &mut base_line)?;
    
    // let label_rect = core::Rect::new(
    //     rect.x, 
    //     rect.y - label_size.height - 8, 
    //     label_size.width + 8, 
    //     label_size.height + 8
    // );
    
    // opencv::imgproc::rectangle(frame, label_rect, core::Scalar::new(255.0, 255.0, 0.0, 0.0), opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;
    
    // let pt = core::Point::new(rect.x, rect.y - 8);
    // opencv::imgproc::put_text(
    //     frame, 
    //     &label, 
    //     pt, 
    //     opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
    //     0.6, 
    //     core::Scalar::new(0.0, 0.0, 0.0, 0.0),
    //     1,
    //     opencv::imgproc::LINE_8, 
    //     false
    // )?;

    Ok(())
}

fn draw_track(frame: &mut Mat, bbox: sort::BBox<sort::Ltwh>, track_id: i32) -> opencv::Result<()> {
    let rect = opencv::core::Rect::new(
        bbox.left() as i32,
        bbox.top() as i32,
        bbox.width() as i32,
        bbox.height() as i32,
    );

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame, 
        rect, 
        core::Scalar::new(0.0, 255.0, 0.0, 0.0), 
        1, 
        opencv::imgproc::LINE_8, 
        0
    )?;

    let label = format!("[{}]", track_id);
    let mut base_line = 0; 
    let label_size = opencv::imgproc::get_text_size(&label, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.6, 1, &mut base_line)?;
    
    let label_rect = core::Rect::new(
        rect.x, 
        rect.y - label_size.height - 8, 
        label_size.width + 8, 
        label_size.height + 8
    );
    
    opencv::imgproc::rectangle(frame, label_rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;
    
    let pt = core::Point::new(rect.x, rect.y - 8);
    opencv::imgproc::put_text(
        frame, 
        &label, 
        pt, 
        opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
        0.6, 
        core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        1,
        opencv::imgproc::LINE_8, 
        false
    )?;

    Ok(())
}

fn main() {
    run().unwrap()
}
