use anyhow::Error;
use grant_object_detector::{Detection, YoloDetector, YoloDetectorConfig};
use deep_sort::{sort, DeepSortConfig, DeepSort, Track};
use ndarray::prelude::*;
use opencv::{
    dnn,
    core::{self, Mat, Scalar, Vector},
    highgui,
    prelude::*,
    videoio,
};

const PALLETE: (i32, i32, i32) = (2047, 32767, 1048575);

fn compute_color_for_labels(label: i32) -> (f64, f64, f64) {
    let c = label * label - label + 1;

    (
        ((PALLETE.0 * c)  % 255) as _,
        ((PALLETE.1 * c)  % 255) as _,
        ((PALLETE.2 * c)  % 255) as _,
    )
}

pub struct YoloDeepSort {
    yolo: YoloDetector,
    deep_sort: DeepSort,
}

impl YoloDeepSort {
    fn new(yolo: &str, reid: &str) -> Result<Self, Error> {
        let device = onnx_model::get_cuda_if_available();
        let mut config = YoloDetectorConfig::new(vec![0, 2, 3, 5, 7]);
        config.confidence_threshold = 0.2;

        Ok(Self {
            yolo: YoloDetector::new(yolo, config, device)?,
            deep_sort: DeepSort::new(DeepSortConfig::new(reid.to_string()))?,
        })
    }

    pub fn detectect(&mut self, frames: &[Mat]) -> Result<Vec<Vec<Detection>>, Error> {
        let (mut frame_width, mut frame_height) = (0i32, 0i32);
        const SIZE: usize = 416; 
        
        let mut inpt = unsafe { Array4::uninitialized([frames.len(), 3, SIZE, SIZE]) };
        for (idx, frame) in frames.iter().enumerate() {
            let fsize = frame.size()?;

            if fsize.width <= 0 {
                continue;
            }

            frame_height = fsize.height;
            frame_width = fsize.width;

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
            let view = aview1(core.data_typed()?).into_shape([3, SIZE, SIZE]).unwrap();
            inpt.index_axis_mut(Axis(0), idx).assign(&view);
        }

        let detections = self.yolo.detect(inpt.view(), frame_width, frame_height)?;

        Ok(detections)
    }

    pub fn track(&mut self, frames: &[Mat], detections: &[Vec<Detection>]) -> Result<&[Track], Error> {
        self.deep_sort.update(frames, detections)?;

        Ok(self.deep_sort.tracks())
    }
}

fn draw_pred(frame: &mut Mat, det: Detection) -> opencv::Result<()> {
    let rect = core::Rect::new(det.xmin, det.ymin, det.xmax - det.xmin, det.ymax - det.ymin);

    //  Draw a bounding box.
    opencv::imgproc::rectangle(
        frame, 
        rect, 
        core::Scalar::new(255.0, 255.0, 0.0, 0.0), 
        1, 
        opencv::imgproc::LINE_8, 
        0
    )?;
    Ok(())
}


fn draw_track(frame: &mut Mat, bbox: sort::BBox<sort::Ltwh>, track_id: i32, color: (f64, f64, f64)) -> opencv::Result<()> {
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
        core::Scalar::new(color.0, color.1, color.2, 0.0), 
        1, 
        opencv::imgproc::LINE_8, 
        0
    )?;

    // let label = format!("[{}]", track_id);
    // let mut base_line = 0; 
    // let label_size = opencv::imgproc::get_text_size(&label, opencv::imgproc::FONT_HERSHEY_SIMPLEX, 0.6, 1, &mut base_line)?;
    
    // let label_rect = core::Rect::new(
    //     rect.x, 
    //     rect.y - label_size.height - 8, 
    //     label_size.width + 8, 
    //     label_size.height + 8
    // );
    
    // opencv::imgproc::rectangle(frame, label_rect, core::Scalar::new(0.0, 255.0, 0.0, 0.0), opencv::imgproc::FILLED, opencv::imgproc::LINE_8, 0)?;
    
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


fn main() -> Result<(), anyhow::Error> {
    let mut tracker = YoloDeepSort::new(
        "/home/andrey/workspace/ssl/yolov4/yolov4_416.onnx", 
        // "/home/andrey/workspace/ssl/reid/onnx_model.onnx",
        "/home/andrey/workspace/ssl/grant/models/model-96.onnx",
        // "/home/andrey/workspace/ssl/deep_sort_pytorch/deep_sort/deep/reid.onnx",
    )?;

    let window = "video capture";
    highgui::named_window(window, 1)?;

    let mut cam = videoio::VideoCapture::from_file("../videoplayback_6.avi", videoio::CAP_ANY)?;  // 0 is the default camera
    // cam.set(videoio::CAP_PROP_POS_FRAMES, 150.0);

    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    
    let mut frames = [core::Mat::default()?];
    loop {
        let begin = std::time::Instant::now();
        cam.read(&mut frames[0])?;

        let detections = tracker.detectect(&frames)?;
        // for d in detections.iter().cloned() {
        //     for d in d {
        //         draw_pred(&mut frames[0], d);
        //     }
        // }

        let tracks = tracker.track(&frames, detections.as_slice())?;
        for t in tracks.iter().filter(|t| t.is_confirmed() && t.time_since_update <= 1) {
            draw_track(&mut frames[0], t.bbox().as_ltwh(), t.track_id, compute_color_for_labels(t.track_id));
        }

        let diff = std::time::Instant::now() - begin;
        let label = format!("{:?}", 1.0 / ((diff.as_millis() as f32) * 0.001));

        opencv::imgproc::put_text(
            &mut frames[0], 
            &label, 
            core::Point::new(30, 30), 
            opencv::imgproc::FONT_HERSHEY_SIMPLEX, 
            0.6, 
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            1,
            opencv::imgproc::LINE_8, 
            false
        )?;

        highgui::imshow(window, &mut frames[0])?;

        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }
    }

    Ok(())
}