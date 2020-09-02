use crate::error::Error;
use onnx_model::{OnnxInferenceModel, OnnxInferenceDevice, TensorView};
use ndarray::prelude::*;

pub struct ImageEncoder {
    model: OnnxInferenceModel, 
}

impl ImageEncoder {
    pub fn new(model_filename: &str, device: OnnxInferenceDevice) -> Result<Self, Error> {
        Ok(Self {
            model: OnnxInferenceModel::new(model_filename, device)?,
        })
    }

    #[inline]
    pub fn input_shape(&self) -> &[i64] {
        let inpt = &self.model.get_input_infos()[0];

        &inpt.shape.dims[1..]
    }

    #[inline]
    pub fn output_shape(&self) -> &[i64] {
        let otpt = &self.model.get_output_infos()[0];

        &otpt.shape.dims[1..]
    }

    pub fn encode_batch(&mut self, in_vals: ArrayView4<'_, f32>) -> std::result::Result<Array2<f32>, Error> {
        let inpt = TensorView::new(in_vals.shape(), in_vals.as_slice().unwrap());
        let otpt = self.model.run(&[inpt])?.pop().unwrap();
        let shape = otpt.dims().clone();
        let features = Array2::from_shape_vec([shape[0] as usize, shape[1] as usize], otpt.to_vec()).unwrap();

        Ok(features)
    }
}
