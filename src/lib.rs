mod ndarrayext;
mod sprsext;
mod sspumv;

use std::result;

pub type Result<T> = result::Result<T, String>;

pub use sspumv::{NdSpline, CubicSmoothingSpline};
