use std::result;

mod errors;
mod ndarrayext;
mod sprsext;
mod sspumv;

pub type Result<T> = result::Result<T, errors::CsapsError>;

pub use sspumv::{NdSpline, CubicSmoothingSpline};
pub use errors::CsapsError;
