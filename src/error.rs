use err_derive::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error(display = "OnnxModel Error: {}", _0)]
    OnnxModelError(onnx_model::error::Error),
}

impl From<onnx_model::error::Error> for Error {
    fn from(err: onnx_model::error::Error) -> Self {
        Self::OnnxModelError(err) 
    }
}