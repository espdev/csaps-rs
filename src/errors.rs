use std::fmt;

/// Enum provides error types
#[derive(PartialEq, Debug)]
pub enum CsapsError {
    /// Any errors when the given input data is invalid
    InvalidInputData(String),

    /// Error occurs when reshape to/from 2-d representation for Y-data has failed
    ReshapeError(String),
}

use self::CsapsError::*;

impl fmt::Display for CsapsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            InvalidInputData(s) => format!("Invalid input: {}", s),
            ReshapeError(s) => s.to_string(),
        };

        s.as_str().fmt(f)
    }
}
