use std::fmt;


#[derive(PartialEq, Debug)]
pub enum CsapsError {
    InvalidInputData(String),
    ReshapeError(String),
    SolveError(String),
}

use self::CsapsError::*;

impl fmt::Display for CsapsError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            InvalidInputData(s) => format!("Invalid input: {}", s),
            ReshapeError(s) => s.to_string(),
            SolveError(s) => s.to_string(),
        };

        s.as_str().fmt(f)
    }
}
