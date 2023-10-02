//! Errors on matrices
#![warn(missing_docs)]

use std::fmt::{Display, Formatter, Result};

#[derive(Debug, PartialEq)]
/// Common Matrix errors that can occur
pub enum MatrixError {
    /// Upon creation of a matrix, this could occur
    MatrixCreationError,
    /// Index out of bound error
    MatrixIndexOutOfBoundsError,
    /// This can only happen on matmul, where if the 2 matrices are not in the form of
    /// (M x N) @ (N x P) then this error will occur.
    MatrixMultiplicationDimensionMismatchError,
    /// Occurs on matrix operations where there is a dimension mismatch between
    /// the two matrices.
    MatrixDimensionMismatchError,
    /// Concatination Error
    MatrixConcatinationError,
    /// If reading matrix from file and an error occurs,
    /// this will be thrown
    MatrixParseError,
    /// Divide by zero
    MatrixDivideByZeroError,
    /// File read error
    MatrixFileReadError(&'static str),
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            MatrixError::MatrixCreationError => {
                write!(f, "There was an error creating the matrix.")
            }
            MatrixError::MatrixIndexOutOfBoundsError => {
                write!(f, "The indexes are out of bounds for the matrix")
            }
            MatrixError::MatrixMultiplicationDimensionMismatchError => {
                write!(
                    f,
                    "The two matrices supplied are not on the form M x N @ N x P"
                )
            }
            MatrixError::MatrixDimensionMismatchError => {
                write!(f, "The matrixs provided are both not on the form M x N")
            }

            MatrixError::MatrixConcatinationError => {
                write!(
                    f,
                    "Matrixs could not be concatinated or extended due to more than 1 dim mismatch"
                )
            }
            MatrixError::MatrixParseError => write!(f, "Failed to parse matrix from file"),
            MatrixError::MatrixDivideByZeroError => write!(f, "Tried to divide by zero"),
            MatrixError::MatrixFileReadError(path) => {
                write!(f, "Could not read file from path: {}", path)
            }
        }
    }
}
