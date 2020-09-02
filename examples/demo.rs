use opencv::{
    dnn,
    core::{self, Mat, Scalar, Vector},
    highgui,
    prelude::*,
    videoio,
};
use deep_sort::{
    deep::ImageEncoder,
    sort,
};
use ndarray::prelude::*;

const CHANNELS: usize = 24;
const CONFIDENCE_THRESHOLD: f32 = 0.6;
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

fn run() -> opencv::Result<()> {
    let mut encoder = ImageEncoder::new("/home/andrey/workspace/ssl/deep_sort_pytorch/deep_sort/deep/reid1.onnx")?;
    let max_cosine_distance = 0.2;
    let nn_budget = 100;
    let max_age = 70;
    let max_iou_distance = 0.2;
    let n_init = 3;
    let kind = sort::NearestNeighborMetricKind::CosineDistance;
    let metric = sort::NearestNeighborDistanceMetric::new(kind, max_cosine_distance, Some(nn_budget));
    let mut tracker = sort::Tracker::new(metric, max_iou_distance, max_age, n_init);

    let model = "/home/andrey/workspace/ssl/yolov3/yolov3.weights";
    let config = "/home/andrey/workspace/ssl/yolov3/yolov3.cfg";
    let framework = "";

    let mut net = dnn::read_net(model, config, framework).unwrap();
    net.set_preferable_backend(dnn::DNN_BACKEND_DEFAULT); 
    net.set_preferable_target(dnn::DNN_TARGET_CPU);

    let layer_names = net.get_layer_names()?;
    let last_layer_id = net.get_layer_id(&layer_names.get(layer_names.len() - 1)?)?;
    let last_layer = net.get_layer(dnn::DictValue::from_i32(last_layer_id)?)?;
    let last_layer_type = last_layer.typ();

    let out_names = net.get_unconnected_out_layers_names().unwrap();

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;  // 0 is the default camera
    
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    
    let mut outs = core::Vector::<core::Mat>::new(); //core::Mat::default()?;

    let mut frame = core::Mat::default()?;
    let mut flag = -1i64;

    loop {
        cam.read(&mut frame)?;

        // flag += 1;
        // if flag % 5 != 0 {
        //     continue;
        // }

        let fsize = frame.size()?;

        if fsize.width <= 0 {
            continue;
        }

        let frame_height = fsize.height;
        let frame_width = fsize.width;

        //  Create a 4D blob from a frame.
        let inp_width = 416;
        let inp_height = 416;
        let blob = dnn::blob_from_image(
            &frame, 
            1.0 / 255.0, 
            core::Size::new(inp_width, inp_height), 
            core::Scalar::new(0., 0., 0., 0.), 
            true, 
            false, 
            core::CV_32F)
            .unwrap();

        //  Run a model
        net.set_input(&blob, "", 1.0, core::Scalar::new(0.,0.,0.,0.));
        net.forward(&mut outs, &out_names).unwrap();

        let fsize = frame.size()?;

        let frame_height = fsize.height;
        let frame_width = fsize.width;
        // let mut objects = vec![];
    
        match last_layer_type.as_str() {
            "Region" => {
                let mut detections = vec![];
                let bboxes = detect(&outs)?;

                for bbox in bboxes {
                    let rect = bbox.cv_rect(frame.cols(), frame.rows());
                    
                    let roi = Mat::roi(&frame, rect)?;
                    let blob = dnn::blob_from_image(
                        &roi, 
                        1.0 / 255.0, 
                        core::Size::new(64, 128), 
                        core::Scalar::new(0., 0., 0., 0.), 
                        true, 
                        false, 
                        core::CV_32F)
                        .unwrap();
                    
                    let code = encoder.encode_batch(&blob)?.get(0)?;
                    let core = code.into_typed::<f32>()?;
                    let feature = arr1(core.data_typed()?);

                    detections.push(sort::Detection {
                        bbox: sort::BBox::ltwh(
                            rect.x as f32, 
                            rect.y as f32, 
                            rect.width as f32, 
                            rect.height as f32
                        ),
                        confidence: bbox.class_confidence, 
                        feature: Some(feature)
                    });

                    draw_pred(&mut frame, bbox)?;
                }

                tracker.predict();
                tracker.update(detections.as_slice());

                for t in tracker.tracks().iter().filter(|t|t.is_confirmed()) {
                    draw_track(&mut frame, t.bbox().as_ltwh(), t.track_id);
                }
            },
    
            _ => panic!("unknown last layer type"),
        }

        highgui::imshow(window, &mut frame)?;

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
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

#[derive(Debug, Clone, Copy)]
pub struct BBox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
    confidence: f32,
    class_index: usize,
    class_confidence: f32,
}

impl BBox {
    pub fn cv_rect(&self, frame_width: i32, frame_height: i32) -> opencv::core::Rect {
        let frame_width_f = frame_width as f32;
        let frame_height_f = frame_height as f32;
        
        let left = ((self.xmin * frame_width_f) as i32).max(0).min(frame_width);
        let top = ((self.ymin * frame_height_f) as i32).max(0).min(frame_height);
        let mut width = (((self.xmax - self.xmin) * frame_width_f) as i32).max(0);
        let mut height = (((self.ymax - self.ymin) * frame_height_f) as i32).max(0);

        if left + width > frame_width {
            width = frame_width - left;
        }

        if top + height > frame_height {
            height = frame_height - top;
        }

        core::Rect::new(
            left,
            top,
            width,
            height,
        )
    }
}

fn detect(pred_events: &Vector<Mat>) -> opencv::Result<Vec<BBox>> {

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
            let pred = pred_event.row(index)?.into_typed::<f32>()?;
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

// Intersection over union of two bounding boxes.
fn iou(b1: &BBox, b2: &BBox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

fn postprocess(frame: &mut Mat, outs: &core::Vector<Mat>, last_layer_type: &str) -> opencv::Result<()> {
    let fsize = frame.size()?;

    let frame_height = fsize.height;
    let frame_width = fsize.width;
    // let mut objects = vec![];

    match last_layer_type {
        "Region" => {
            // let bboxes = detect(&outs)?;

            // for bbox in bboxes {
            //     draw_pred(frame, bbox);
            // }
        },

        _ => panic!("unknown last layer type"),
    }

    // classIds = []
    // confidences = []
    // boxes = []
    // if lastLayer.type == 'DetectionOutput':
    //     # Network produces output blob with a shape 1x1xNx7 where N is a number of
    //     # detections and an every detection is a vector of values
    //     # [batchId, classId, confidence, left, top, right, bottom]
    //     for out in outs:
    //         for detection in out[0, 0]:
    //             confidence = detection[2]
    //             if confidence > confThreshold:
    //                 left = int(detection[3])
    //                 top = int(detection[4])
    //                 right = int(detection[5])
    //                 bottom = int(detection[6])
    //                 width = right - left + 1
    //                 height = bottom - top + 1
    //                 if width <= 2 or height <= 2:
    //                     left = int(detection[3] * frameWidth)
    //                     top = int(detection[4] * frameHeight)
    //                     right = int(detection[5] * frameWidth)
    //                     bottom = int(detection[6] * frameHeight)
    //                     width = right - left + 1
    //                     height = bottom - top + 1
    //                 classIds.append(int(detection[1]) - 1)  # Skip background label
    //                 confidences.append(float(confidence))
    //                 boxes.append([left, top, width, height])
    // elif lastLayer.type == 'Region':
    //     # Network produces output blob with a shape NxC where N is a number of
    //     # detected objects and C is a number of classes + 4 where the first 4
    //     # numbers are [center_x, center_y, width, height]
    //     for out in outs:
    //         for detection in out:
    //             scores = detection[5:]
    //             classId = np.argmax(scores)
    //             confidence = scores[classId]
    //             if confidence > confThreshold:
    //                 center_x = int(detection[0] * frameWidth)
    //                 center_y = int(detection[1] * frameHeight)
    //                 width = int(detection[2] * frameWidth)
    //                 height = int(detection[3] * frameHeight)
    //                 left = int(center_x - width / 2)
    //                 top = int(center_y - height / 2)
    //                 classIds.append(classId)
    //                 confidences.append(float(confidence))
    //                 boxes.append([left, top, width, height])
    // else:
    //     print('Unknown output layer type: ' + lastLayer.type)
    //     exit()

    // # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    // # or NMS is required if number of outputs > 1
    // if len(outNames) > 1 or lastLayer.type == 'Region' and args.backend != cv.dnn.DNN_BACKEND_OPENCV:
    //     indices = []
    //     classIds = np.array(classIds)
    //     boxes = np.array(boxes)
    //     confidences = np.array(confidences)
    //     unique_classes = set(classIds)
    //     for cl in unique_classes:
    //         class_indices = np.where(classIds == cl)[0]
    //         conf = confidences[class_indices]
    //         box  = boxes[class_indices].tolist()
    //         nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
    //         nms_indices = nms_indices[:, 0] if len(nms_indices) else []
    //         indices.extend(class_indices[nms_indices])
    // else:
    //     indices = np.arange(0, len(classIds))

    // for i in indices:
    //     box = boxes[i]
    //     left = box[0]
    //     top = box[1]
    //     width = box[2]
    //     height = box[3]
    //     drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    Ok(())
}

fn main() {
    run().unwrap()
}
