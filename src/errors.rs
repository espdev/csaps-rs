use ndarray::ShapeError;
use thiserror::Error;

/// Enum provides error types
#[derive(Error, Debug)]
pub enum CsapsError {
    /// Any errors when the given input data is invalid
    #[error("Invalid input: {0}")]
    InvalidInputData(String),

    /// Error occurs when reshape from 2-d representation for n-d data has failed
    #[error("Cannot reshape 2-d array with shape {input_shape:?} \
             to {}-d array with shape {output_shape:?} by axis {axis}. Error: {source}",
            output_shape.len())]
    ReshapeFrom2d {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        axis: usize,
        #[source]
        source: ShapeError,
    },

    /// Error occurs when reshape to 2-d representation for n-d data has failed
    #[error("Cannot reshape {}-d array with shape {input_shape:?} by axis {axis} \
             to 2-d array with shape {output_shape:?}. Error: {source}",
            input_shape.len())]
    ReshapeTo2d {
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        axis: usize,
        #[source]
        source: ShapeError,
    },
}
