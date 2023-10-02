//! Makin with matrices in rust easier!
//!
//! For now, only basic operations are allowed, but more are to be added
//!
//! This file is sub 1500 lines and acts as the core file

mod helper;
mod optim;

use helper::*;

use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fmt::{Debug, Display},
    fs,
    marker::PhantomData,
    ops::Div,
    str::FromStr,
};

use itertools::{iproduct, Itertools};
use num_traits::{pow, real::Real, sign::abs, Float};
use rand::Rng;
use rayon::prelude::*;
use std::iter::Sum;

use crate::{at, LinAlgFloats, MatrixElement, MatrixError, SparseMatrix};

/// Shape represents the dimension size
/// of the matrix as a tuple of usize
pub type Shape = (usize, usize);

/// Helper method to swap to usizes

#[derive(Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
/// General dense matrix
pub struct Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Vector containing all data
    data: Vec<T>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Error for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Send for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Sync for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

impl<'a, T> FromStr for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the input string and construct the matrix dynamically
        let v: Vec<T> = s
            .trim()
            .lines()
            .map(|l| {
                l.split_whitespace()
                    .map(|num| num.parse::<T>().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .into_iter()
            .flatten()
            .collect();

        let rows = s.trim().lines().count();
        let cols = s.trim().lines().nth(0).unwrap().split_whitespace().count();

        Ok(Self::new(v, (rows, cols)).unwrap())
    }
}

impl<'a, T> Display for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[");

        // Large matrices
        if self.nrows > 10 || self.ncols > 10 {
            write!(f, "...");
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if i == 0 {
                    write!(f, "{:.4} ", self.get(i, j).unwrap());
                } else {
                    write!(f, " {:.4}", self.get(i, j).unwrap());
                }
            }
            // Print ] on same line if youre at the end
            if i == self.nrows - 1 {
                break;
            }
            writeln!(f);
        }
        writeln!(f, "], dtype={}", std::any::type_name::<T>())
    }
}

impl<'a, T> Default for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Represents a default identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f32> = Matrix::default();
    ///
    /// assert_eq!(matrix.size(), 9);
    /// assert_eq!(matrix.shape(), (3,3));
    /// ```
    fn default() -> Self {
        Self::eye(3)
    }
}

/// Printer functions for the matrix
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Prints out the matrix with however many decimals you want
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<i32> = Matrix::eye(2);
    /// matrix.print(4);
    ///
    /// ```
    pub fn print(&self, decimals: usize) {
        print!("[");

        // Large matrices
        if self.nrows > 10 || self.ncols > 10 {
            print!("...");
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if i == 0 {
                    print!(
                        "{val:.dec$} ",
                        dec = decimals,
                        val = self.get(i, j).unwrap()
                    );
                } else {
                    print!(
                        " {val:.dec$}",
                        dec = decimals,
                        val = self.get(i, j).unwrap()
                    );
                }
            }
            // Print ] on same line if youre at the end
            if i == self.nrows - 1 {
                break;
            }
            println!();
        }
        println!("], dtype={}", std::any::type_name::<T>());
    }

    /// Calculates sparsity of a given Matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mat: Matrix<f32> = Matrix::eye(2);
    ///
    /// assert_eq!(mat.sparsity(), 0.5);
    /// ```
    #[inline(always)]
    pub fn sparsity(&'a self) -> f64 {
        self.count_where(|&e| e == T::zero()) as f64 / self.size() as f64
    }

    /// Returns the shape of a matrix represented as  
    /// (usize, usize)
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mat: Matrix<f32> = Matrix::eye(4);
    ///
    /// assert_eq!(mat.shape(), (4,4));
    /// ```
    pub fn shape(&self) -> Shape {
        (self.nrows, self.ncols)
    }
}

/// Implementations of all creatins of matrices
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Creates a new matrix from a vector and the shape you want.
    /// Will default init if it does not work
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![1.0,2.0,3.0,4.0], (2,2)).unwrap();
    ///
    /// assert_eq!(matrix.size(), 4);
    /// assert_eq!(matrix.shape(), (2,2));
    /// ```
    pub fn new(data: Vec<T>, shape: Shape) -> Result<Self, MatrixError> {
        if shape.0 * shape.1 != data.len() {
            return Err(MatrixError::MatrixCreationError.into());
        }

        Ok(Self {
            data,
            nrows: shape.0,
            ncols: shape.1,
            _lifetime: PhantomData::default(),
        })
    }

    /// Initializes a matrix with the same value
    /// given from parameter 'value'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(4f32, (1,2));
    ///
    /// assert_eq!(matrix.get_vec(), vec![4f32,4f32]);
    /// assert_eq!(matrix.shape(), (1,2));
    /// ```
    pub fn init(value: T, shape: Shape) -> Self {
        Self::from_shape(value, shape)
    }

    /// Returns an eye matrix which for now is the same as the
    /// identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f32> = Matrix::eye(2);
    ///
    /// assert_eq!(matrix.get_vec(), vec![1f32, 0f32, 0f32, 1f32]);
    /// assert_eq!(matrix.shape(), (2,2));
    /// ```
    pub fn eye(size: usize) -> Self {
        let mut data: Vec<T> = vec![T::zero(); size * size];

        (0..size).for_each(|i| data[i * size + i] = T::one());

        // Safe to do since the library is setting the size
        Self::new(data, (size, size)).unwrap()
    }

    /// Produce an identity matrix in the same shape as
    /// an already existent matrix
    ///
    /// Examples
    ///
    /// ```
    ///
    ///
    /// ```
    pub fn eye_like(matrix: &Self) -> Self {
        Self::eye(matrix.nrows)
    }

    /// Identity is same as eye, just for nerds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<i64> = Matrix::identity(2);
    ///
    /// assert_eq!(matrix.get_vec(), vec![1i64, 0i64, 0i64, 1i64]);
    /// assert_eq!(matrix.shape(), (2,2));
    /// ```
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Tries to create a matrix from a slize and shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let s = vec![1f32, 2f32, 3f32, 4f32];
    /// let matrix = Matrix::from_slice(&s, (4,1)).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (4,1));
    /// ```
    pub fn from_slice(arr: &[T], shape: Shape) -> Result<Self, MatrixError> {
        if shape.0 * shape.1 != arr.len() {
            return Err(MatrixError::MatrixCreationError.into());
        }

        Ok(Self::new(arr.to_owned(), shape).unwrap())
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f64> = Matrix::zeros((4,1));
    ///
    /// assert_eq!(matrix.shape(), (4,1));
    /// assert_eq!(matrix.get(0,0).unwrap(), 0f64);
    /// assert_eq!(matrix.size(), 4);
    /// ```
    pub fn zeros(shape: Shape) -> Self {
        Self::from_shape(T::zero(), shape)
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f64> = Matrix::ones((4,1));
    ///
    /// assert_eq!(matrix.shape(), (4,1));
    /// assert_eq!(matrix.get(0,0).unwrap(), 1f64);
    /// assert_eq!(matrix.size(), 4);
    /// ```
    pub fn ones(shape: Shape) -> Self {
        Self::from_shape(T::one(), shape)
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<i8> = Matrix::default();
    /// let matrix2 = Matrix::zeros_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape(), matrix1.shape());
    /// assert_eq!(matrix2.get(0,0).unwrap(), 0i8);
    /// ```
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(T::zero(), other.shape())
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<i64> = Matrix::default();
    /// let matrix2 = Matrix::ones_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape(), matrix1.shape());
    /// assert_eq!(1i64, matrix2.get(0,0).unwrap());
    /// ```
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(T::one(), other.shape())
    }

    /// Creates a matrix where all values are random between 0 and 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<f32> = Matrix::default();
    /// let matrix2 = Matrix::random_like(&matrix1);
    ///
    /// assert_eq!(matrix1.shape(), matrix2.shape());
    ///
    ///
    /// ```
    pub fn random_like(matrix: &Self) -> Self {
        Self::randomize_range(T::zero(), T::one(), matrix.shape())
    }

    /// Creates a matrix where all values are random between start..=end.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::randomize_range(1f32, 2f32, (2,3));
    /// let elem = matrix.get(1,1).unwrap();
    ///
    /// assert_eq!(matrix.shape(), (2,3));
    /// //assert!(elem >= 1f32 && 2f32 <= elem);
    /// ```
    pub fn randomize_range(start: T, end: T, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data: Vec<T> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        // Safe because shape doesn't have to match data from a user
        Self::new(data, shape).unwrap()
    }

    /// Creates a matrix where all values are random between 0..=1.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f64> = Matrix::randomize((2,3));
    ///
    /// assert_eq!(matrix.shape(), (2,3));
    /// ```
    pub fn randomize(shape: Shape) -> Self {
        Self::randomize_range(T::zero(), T::one(), shape)
    }

    /// Parses from file, but will return a default matrix if nothing is given
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// // let m: Matrix<f32> = Matrix::from_file("../../test.txt").unwrap();
    ///
    /// // m.print(4);
    /// ```
    pub fn from_file(path: &'static str) -> Result<Self, MatrixError> {
        let data =
            fs::read_to_string(path).map_err(|_| MatrixError::MatrixFileReadError(path).into())?;

        data.parse::<Self>()
            .map_err(|_| MatrixError::MatrixParseError.into())
    }

    /// Constructs a new dense matrix from a sparse one.
    ///
    /// This transfesrs ownership as well!
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::{Matrix, SparseMatrix};
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// let matrix = Matrix::from_sparse(sparse);
    ///
    /// assert_eq!(matrix.shape(), (3,3));
    /// assert_eq!(matrix.at(0,0), 1);
    /// ```
    pub fn from_sparse(sparse: SparseMatrix<'a, T>) -> Self {
        let mut mat = Self::zeros(sparse.shape());

        for (&idx, &val) in sparse.data.iter() {
            mat.set(val, idx);
        }

        mat
    }

    /// Helper function to create matrices
    fn from_shape(value: T, shape: Shape) -> Self {
        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data = vec![value; len];

        Self::new(data, shape).unwrap()
    }
}

/// Enum for specifying which dimension / axis to work with
pub enum Dimension {
    /// Row is defined as 0
    Row = 0,
    /// Col is defined as 1
    Col = 1,
}

/// Regular matrix methods that are not operating math on them
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement + Div<Output = T> + Sum<T>,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Reshapes a matrix if possible.
    /// If the shapes don't match up, the old shape will be retained
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.reshape(3,2);
    ///
    /// assert_eq!(matrix.shape(), (3,2));
    /// ```
    pub fn reshape(&mut self, nrows: usize, ncols: usize) {
        if nrows * ncols != self.size() {
            eprintln!("Err: Can not reshape.. Keeping old dimensions for now");
            return;
        }

        self.nrows = nrows;
        self.ncols = ncols;
    }

    /// Get the total size of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.nrows * self.ncols
    }

    ///  Gets element based on is and js
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5f32, (2,3));
    ///
    /// assert_eq!(matrix.get(0,1).unwrap(), 10.5f32);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        let idx = at!(i, j, self.ncols);

        if idx >= self.size() {
            return None;
        }

        Some(self.at(i, j))
    }

    ///  Gets element based on is and js, but will
    ///  panic if indexes are out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let val = 10.5;
    ///
    /// let matrix = Matrix::init(val, (2,3));
    ///
    /// assert_eq!(matrix.at(1,2), val);
    /// ```
    #[inline(always)]
    pub fn at(&self, i: usize, j: usize) -> T {
        self.data[at!(i, j, self.ncols)]
    }

    ///  Gets a piece of the matrix out as a vector
    ///
    ///  If some indeces are out of bounds, the vec up until that point
    ///  will be returned
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let slice = matrix.get_vec_slice((1,1), (2,2));
    ///
    /// assert_eq!(slice, vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_vec_slice(&self, start_idx: Shape, size: Shape) -> Vec<T> {
        let (start_row, start_col) = start_idx;
        let (dx, dy) = size;

        iproduct!(start_row..start_row + dy, start_col..start_col + dx)
            .filter_map(|(i, j)| self.get(i, j))
            .collect()
    }

    /// Gets you the whole entire matrix as a vector
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let slice = matrix.get_vec_slice((1,1), (2,2));
    ///
    /// assert_eq!(slice, vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    ///  Gets a piece of the matrix out as a matrix
    ///
    ///  If some indeces are out of bounds, unlike `get_vec_slice`
    ///  this function will return an IndexOutOfBoundsError
    ///  and will not return data
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let sub_matrix = matrix.get_sub_matrix((1,1), (2,2)).unwrap();
    ///
    /// assert_eq!(sub_matrix.get_vec(), vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_sub_matrix(&self, start_idx: Shape, size: Shape) -> Result<Self, MatrixError> {
        let (start_row, start_col) = start_idx;
        let (dx, dy) = size;

        let data = iproduct!(start_row..start_row + dy, start_col..start_col + dx)
            .filter_map(|(i, j)| self.get(i, j))
            .collect();

        return match Self::new(data, size) {
            Ok(a) => Ok(a),
            Err(_) => Err(MatrixError::MatrixIndexOutOfBoundsError.into()),
        };
    }

    /// Concat two mtrices on a dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let matrix2 = Matrix::init(10.5, (1,4));
    ///
    /// let res = matrix.concat(&matrix2, Dimension::Row).unwrap();
    ///
    /// assert_eq!(res.shape(), (5,4));
    /// ```
    pub fn concat(&self, other: &Self, dim: Dimension) -> Result<Self, MatrixError> {
        match dim {
            Dimension::Row => {
                if self.ncols != other.ncols {
                    return Err(MatrixError::MatrixConcatinationError.into());
                }

                let mut new_data = self.data.clone();

                new_data.extend(other.data.iter());

                let nrows = self.nrows + other.nrows;
                let shape = (nrows, self.ncols);

                return Ok(Self::new(new_data, shape).unwrap());
            }

            Dimension::Col => {
                if self.nrows != other.nrows {
                    return Err(MatrixError::MatrixConcatinationError.into());
                }

                let mut new_data: Vec<T> = Vec::new();

                let take_self = self.ncols;
                let take_other = other.ncols;

                for (idx, _) in self.data.iter().step_by(take_self).enumerate() {
                    // Add from self, then other
                    let row = (idx / take_self) * take_self;
                    new_data.extend(self.data.iter().skip(row).take(take_self));
                    new_data.extend(other.data.iter().skip(row).take(take_other));
                }

                let ncols = self.ncols + other.ncols;
                let shape = (self.nrows, ncols);

                return Ok(Self::new(new_data, shape).unwrap());
            }
        };
    }

    // TODO: Add option to transpose to be able to extend
    // Doens't change anything if dimension mismatch

    /// Extend a matrix with another on a dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let mut matrix = Matrix::init(10.5, (4,4));
    /// let matrix2 = Matrix::init(10.5, (4,1));
    ///
    /// matrix.extend(&matrix2, Dimension::Col);
    ///
    /// assert_eq!(matrix.shape(), (4,5));
    /// ```
    pub fn extend(&mut self, other: &Self, dim: Dimension) {
        match dim {
            Dimension::Row => {
                if self.ncols != other.ncols {
                    eprintln!("Error: Dimension mismatch");
                    return;
                }

                self.data.extend(other.data.iter());

                self.nrows += other.nrows;
            }

            Dimension::Col => {
                if self.nrows != other.nrows {
                    eprintln!("Error: Dimension mismatch");
                    return;
                }

                let mut new_data: Vec<T> = Vec::new();

                let take_self = self.ncols;
                let take_other = other.ncols;

                for (idx, _) in self.data.iter().step_by(take_self).enumerate() {
                    // Add from self, then other
                    let row = (idx / take_self) * take_self;
                    new_data.extend(self.data.iter().skip(row).take(take_self));
                    new_data.extend(other.data.iter().skip(row).take(take_other));
                }

                self.ncols += other.ncols;
            }
        };
    }

    ///  Sets element based on is and js
    ///
    ///  Sets nothing if you;re out of bounds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.set(11.5, (1, 2));
    ///
    /// assert_eq!(matrix.get(1,2).unwrap(), 11.5);
    /// ```
    pub fn set(&mut self, value: T, idx: Shape) {
        let idx = at!(idx.0, idx.1, self.ncols);

        if idx >= self.size() {
            eprintln!("Error: Index out of bounds. Not setting value.");
            return;
        }

        self.data[idx] = value;
    }

    ///  Sets many elements based on vector of indeces
    ///
    ///  For indexes out of bounds, nothing is set
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.set_many(vec![(1,2), (1,1)], 11.5);
    ///
    /// assert_eq!(matrix.get(1,2).unwrap(), 11.5);
    /// assert_eq!(matrix.get(1,1).unwrap(), 11.5);
    /// assert_eq!(matrix.get(0,1).unwrap(), 10.5);
    /// ```
    pub fn set_many(&mut self, idx_list: Vec<Shape>, value: T) {
        idx_list.iter().for_each(|&idx| self.set(value, idx));
    }

    /// Sets all elements of a matrix in a 1d range.
    ///
    /// The range is inclusive to stop, and will panic
    /// if any indexes are out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.set_range(0, 3, 11.5);
    ///
    /// assert_eq!(matrix.get(0,2).unwrap(), 11.5);
    /// assert_eq!(matrix.get(0,1).unwrap(), 11.5);
    /// assert_eq!(matrix.get(1,1).unwrap(), 10.5);
    /// ```
    pub fn set_range(&mut self, start: usize, stop: usize, value: T) {
        (start..=stop).for_each(|i| self.data[i] = value);
    }

    /// Calculates the (row, col) for a matrix by a single index
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,2));
    /// let inv = matrix.one_to_2d_idx(1);
    ///
    /// assert_eq!(inv, (0,1));
    /// ```
    pub fn one_to_2d_idx(&self, idx: usize) -> Shape {
        let row = idx / self.ncols;
        let col = idx % self.ncols;

        (row, col)
    }

    /// Finds maximum element in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.max(), 10.5);
    /// ```
    pub fn max(&self) -> T {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds minimum element in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.set(1.0, (0,2));
    ///
    /// assert_eq!(matrix.min(), 1.0);
    /// ```
    pub fn min(&self) -> T {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds position in matrix where value is highest.
    /// Restricted to find this across a row or column
    /// in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(1.0, (3,3));
    /// matrix.set(15.0, (0,2));
    ///
    /// ```
    fn argmax(&self, rowcol: usize, dimension: Dimension) -> Option<Shape> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.nrows - 1 {
                    return None;
                }

                let mut highest: T = T::one();
                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol * self.ncols)
                    .take(self.ncols)
                {
                    if *elem >= highest {
                        i = idx;
                    }
                }

                Some(self.one_to_2d_idx(i))
            }

            Dimension::Col => {
                if rowcol >= self.ncols - 1 {
                    return None;
                }

                let mut highest: T = T::one();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol)
                    .step_by(self.ncols)
                {
                    if *elem >= highest {
                        i = idx;
                    }
                }

                Some(self.one_to_2d_idx(i))
            }
        }
    }

    /// Finds position in matrix where value is lowest.
    /// Restricted to find this across a row or column
    /// in the matrix.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(10.5, (3,3));
    /// matrix.set(1.0, (0,1));
    ///
    /// // assert_eq!(matrix.argmin(1, Dimension::Col), Some(1));
    /// ```
    fn argmin(&self, rowcol: usize, dimension: Dimension) -> Option<Shape> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.nrows - 1 {
                    return None;
                }

                let mut lowest: T = T::zero();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol * self.ncols)
                    .take(self.ncols)
                {
                    if *elem < lowest {
                        i = idx;
                    }
                }

                Some(self.one_to_2d_idx(i))
            }

            Dimension::Col => {
                if rowcol >= self.ncols - 1 {
                    return None;
                }

                let mut lowest: T = T::zero();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol)
                    .step_by(self.ncols)
                {
                    if *elem <= lowest {
                        i = idx;
                    }
                }

                Some(self.one_to_2d_idx(i))
            }
        }
    }

    /// Finds total sum of matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.cumsum(), 40.0);
    /// ```
    pub fn cumsum(&self) -> T {
        if self.size() == 0 {
            return T::zero();
        }

        self.data.par_iter().copied().sum()
    }

    /// Multiplies  all elements in matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.cumprod(), 10000.0);
    /// ```
    pub fn cumprod(&self) -> T {
        if self.size() == 0 {
            return T::zero();
        }

        self.data.par_iter().copied().product()
    }

    /// Gets the average of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.avg(), 10.0);
    /// ```
    pub fn avg(&self) -> T {
        self.data.par_iter().copied().sum::<T>() / self.size().to_string().parse::<T>().unwrap()
    }

    /// Gets the mean of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.mean(), 10.0);
    /// ```
    pub fn mean(&self) -> T {
        self.avg()
    }

    /// Gets the median of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![1.0, 4.0, 6.0, 5.0], (2,2)).unwrap();
    ///
    /// assert!(matrix.median() >= 4.45 && matrix.median() <= 4.55);
    /// ```
    pub fn median(&self) -> T {
        if self.size() == 1 {
            return self.at(0, 0);
        }

        match self.data.len() % 2 {
            0 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .skip(half - 1)
                    .take(2)
                    .copied()
                    .sum::<T>()
                    / (T::one() + T::one())
            }
            1 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .nth(half)
                    .copied()
                    .unwrap()
            }
            _ => unreachable!(),
        }
    }

    /// Sums up elements over given axis and dimension.
    /// Will return 0 if you're out of bounds
    ///
    /// sum(2, Dimension::Col) means summig up these ones
    ///
    /// [ 10 10 (10) 10 10
    ///   10 10 (10) 10 10
    ///   10 10 (10) 10 10
    ///   10 10 (10) 10 10
    ///   10 10 (10) 10 10 ]
    ///
    ///   = 10 * 5 = 50
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10f32, (5,5));
    ///
    /// assert_eq!(matrix.sum(0, Dimension::Row), 50.0);
    /// assert_eq!(matrix.sum(3, Dimension::Col), 50.0);
    /// ```
    pub fn sum(&self, rowcol: usize, dimension: Dimension) -> T {
        // TODO: Add out of bounds options
        if self.size() == 1 {
            return self.at(0, 0);
        }

        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.ncols)
                .take(self.ncols)
                .copied()
                .sum(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.ncols)
                .copied()
                .sum(),
        }
    }

    /// Prods up elements over given rowcol and dimension
    /// Will return 1 if you're out of bounds.
    ///
    /// See `sum` for example on how this is calculated
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.prod(0, Dimension::Row), 100.0);
    /// assert_eq!(matrix.prod(0, Dimension::Col), 100.0);
    /// ```
    pub fn prod(&self, rowcol: usize, dimension: Dimension) -> T {
        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.ncols)
                .take(self.ncols)
                .copied()
                .product(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.ncols)
                .copied()
                .product(),
        }
    }
}

/// Linalg on floats
impl<'a, T> LinAlgFloats<'a, T> for Matrix<'a, T>
where
    T: MatrixElement + Float + 'a,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Takes the logarithm of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    ///
    /// let matrix = Matrix::init(10.0, (2,2));
    /// let result = matrix.log(10.0);
    ///
    /// assert_eq!(result.all(|&e| e == 1.0), true);
    ///
    /// ```
    fn log(&self, base: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.log(base)).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Takes the natural logarithm of each element in a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF64;
    ///
    /// let matrix: Matrix<f64> = Matrix::init(EF64, (2,2));
    ///
    /// let res = matrix.ln();
    /// ```
    fn ln(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.ln()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Takes the square root of each element in a matrix.
    /// If some elements are negative, these will be kept the same
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    ///
    /// let matrix: Matrix<f64> = Matrix::init(9.0, (3,3));
    ///
    /// let res = matrix.sqrt();
    ///
    /// assert_eq!(res.all(|&e| e == 3.0), true);
    /// ```
    fn sqrt(&self) -> Self {
        let data: Vec<T> = self
            .data
            .par_iter()
            .map(|&e| if e > T::zero() { e.sqrt() } else { e })
            .collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets sin of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    ///
    /// let matrix = Matrix::init(1.0, (2,2));
    ///
    /// let res = matrix.sin();
    /// ```
    fn sin(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.sin()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets cos of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// let res = matrix.cos();
    /// ```
    fn cos(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.cos()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets tan of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// let res = matrix.tan();
    /// ```
    fn tan(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.tan()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets sinh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// let res = matrix.sinh();
    /// ```
    fn sinh(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.sinh()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets cosh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// let res = matrix.cosh();
    /// ```
    fn cosh(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.cosh()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Gets tanh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// let res = matrix.tanh();
    /// ```
    fn tanh(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.tanh()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Find the eigenvale of a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    ///
    /// ```
    fn get_eigenvalues(&self) -> Option<Vec<T>> {
        todo!()
    }

    /// Find the eigenvectors
    fn get_eigenvectors(&self) -> Option<Vec<T>> {
        unimplemented!()
    }
}

/// trait MatrixLinAlg contains all common Linear Algebra functions to be
/// performed on matrices
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Adds one matrix to another
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(10.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.add(&matrix2).unwrap().get(0,0).unwrap(), 20.0);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Ok(Self::new(data, self.shape()).unwrap())
    }

    /// Subtracts one matrix from another
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(10.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.sub(&matrix2).unwrap().get(1,0).unwrap(), 0.0);
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x - y)
            .collect();

        Ok(Self::new(data, self.shape()).unwrap())
    }

    /// Subtracts one array from another and returns the absolute value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(10.0f32, (2,2));
    /// let matrix2 = Matrix::init(15.0f32, (2,2));
    ///
    /// assert_eq!(matrix1.sub_abs(&matrix2).unwrap().get(0,0).unwrap(), 5.0);
    /// ```
    pub fn sub_abs(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| if x > y { x - y } else { y - x })
            .collect_vec();

        Ok(Self::new(data, self.shape()).unwrap())
    }

    /// Dot product of two matrices
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.mul(&matrix2).unwrap().get(0,0).unwrap(), 200.0);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x * y)
            .collect_vec();

        Ok(Self::new(data, self.shape()).unwrap())
    }

    /// Dot product of two matrices
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.dot(&matrix2).unwrap().get(0,0).unwrap(), 200.0);
    /// ```
    pub fn dot(&self, other: &Self) -> Result<Self, MatrixError> {
        self.mul(other)
    }

    /// Bad handling of zero div
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.div(&matrix2).unwrap().get(0,0).unwrap(), 2.0);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        if other.any(|e| e == &T::zero()) {
            return Err(MatrixError::MatrixDivideByZeroError.into());
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x / y)
            .collect_vec();

        Ok(Self::new(data, self.shape()).unwrap())
    }

    /// Negates every value in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, LinAlgFloats};
    ///
    /// let matrix = Matrix::<f32>::ones((2,2));
    ///
    /// let negated = matrix.neg();
    ///
    /// assert_eq!(negated.all(|&e| e == -1.0), true);
    /// ```
    pub fn neg(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.neg()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Adds a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.add_val(value).get(0,0).unwrap(), 22.0);
    /// ```
    pub fn add_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e + val).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Substracts a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.sub_val(value).get(0,0).unwrap(), 18.0);
    /// ```
    pub fn sub_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e - val).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Multiplies a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.mul_val(value).get(0,0).unwrap(), 40.0);
    /// ```
    pub fn mul_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e * val).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Divides a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// let result_mat = matrix.div_val(value);
    ///
    /// assert_eq!(result_mat.get(0,0).unwrap(), 10.0);
    /// ```
    pub fn div_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e / val).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Pows each value in a matrix by val times
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    ///
    /// let result_mat = matrix.pow(2);
    ///
    /// assert_eq!(result_mat.get_vec(), vec![4.0, 4.0, 4.0, 4.0]);
    /// ```
    pub fn pow(&self, val: usize) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| pow(e, val)).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Takes the absolute values of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(-20.0, (2,2));
    ///
    /// let res = matrix.abs();
    ///
    /// assert_eq!(res.all(|&e| e == 20.0), true);
    /// ```
    pub fn abs(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.abs()).collect();

        Self::new(data, self.shape()).unwrap()
    }

    /// Multiply a matrix with itself n number of times.
    /// This is done by performing a matrix multiplication
    /// several time on self and the result of mat.exp(i-1).
    ///
    /// If matrix is not in form NxN, this function returns None
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let a = Matrix::<i32>::init(2, (2,2));
    ///
    /// let res = a.exp(3).unwrap();
    ///
    /// assert_eq!(res.all(|&e| e == 32), true);
    /// ```
    pub fn exp(&self, n: usize) -> Option<Self> {
        if self.nrows != self.ncols {
            return None;
        }

        let mut res = self.clone();

        (0..n - 1).for_each(|_| res = res.matmul(self).unwrap());

        Some(res)
    }

    /// Adds a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.add_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0).unwrap(), 22.0);
    /// ```
    pub fn add_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += *b);
    }

    /// Subtracts a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.sub_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0).unwrap(), 18.0);
    /// ```
    pub fn sub_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a -= *b);
    }

    /// Multiplies a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.mul_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0).unwrap(), 40.0);
    /// ```
    pub fn mul_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a *= *b);
    }

    /// Divides a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.div_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0).unwrap(), 10.0);
    /// ```
    pub fn div_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a /= *b);
    }

    /// Abs matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    ///
    /// matrix.abs_self()
    ///
    /// // assert_eq!(matrix1.get(0,0).unwrap(), 22.0);
    /// ```
    pub fn abs_self(&mut self) {
        self.data.par_iter_mut().for_each(|e| *e = abs(*e))
    }

    /// Adds a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.add_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 22.0);
    /// ```
    pub fn add_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e += val);
    }

    /// Subtracts a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.sub_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 18.0);
    /// ```
    pub fn sub_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e -= val);
    }

    /// Mults a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.mul_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 40.0);
    /// ```
    pub fn mul_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e *= val);
    }

    /// Divs a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.div_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 10.0);
    /// ```
    pub fn div_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e /= val);
    }

    /// Transposed matrix multiplications
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix1 = Matrix::init(2.0, (2,4));
    /// let matrix2 = Matrix::init(2.0, (4,2));
    ///
    /// let result = matrix1.matmul(&matrix2).unwrap();
    ///
    /// assert_eq!(result.get(0,0).unwrap(), 16.0);
    /// assert_eq!(result.shape(), (2,2));
    /// ```
    pub fn matmul(&self, other: &Self) -> Result<Self, MatrixError> {
        // assert M N x N P
        if self.ncols != other.nrows {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        Ok(self.matmul_helper(other))
    }

    /// Shorthand method for matmul
    pub fn mm(&self, other: &Self) -> Result<Self, MatrixError> {
        self.matmul(other)
    }

    /// Get's the determinat of a N x N matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mat: Matrix<i32> = Matrix::new(vec![1,3,5,9,1,3,1,7,4,3,9,7,5,2,0,9], (4,4)).unwrap();
    ///
    ///
    /// let res = mat.determinant().unwrap();
    ///
    /// assert_eq!(res, -376);
    /// ```
    pub fn determinant(&self) -> Option<T> {
        if self.nrows != self.ncols {
            return None;
        }

        Some(self.determinant_helper())
    }

    /// Shorthand call for `determinant`
    pub fn det(&self) -> Option<T> {
        self.determinant()
    }

    /// Finds the inverse of a matrix if possible
    ///
    /// Definition: AA^-1 = A^-1A = I
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![4,7,2,6], (2,2)).unwrap();
    ///
    /// // let inverse  = matrix.inverse();
    ///
    /// ```
    pub fn inverse(&self) -> Option<Self> {
        if self.shape() != (2, 2) {
            eprintln!("Function not implemented for inverse on larger matrices yet!");
            return None;
        }
        if self.nrows != self.ncols {
            eprintln!("Oops");
            return None;
        }

        if self.determinant().unwrap() == T::zero() {
            return None;
        }

        let a = self.at(0, 0);
        let b = self.at(0, 1);
        let c = self.at(1, 0);
        let d = self.at(1, 1);

        let mut mat = Self::new(vec![d, -b, -c, a], self.shape()).unwrap();

        mat.mul_val_self(T::one() / (a * d - b * c));

        return Some(mat);

        // let mut inverse = Self::zeros_like(self);
        //
        // let identity_mat = Self::eye_like(self);
        //
        // Some(inverse)
    }

    /// Transpose a matrix in-place
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    /// matrix.transpose();
    ///
    /// assert_eq!(matrix.shape(), (100,2));
    /// ```
    pub fn transpose(&mut self) {
        for i in 0..self.nrows {
            for j in (i + 1)..self.ncols {
                let lhs = at!(i, j, self.ncols);
                let rhs = at!(j, i, self.nrows);
                self.data.swap(lhs, rhs);
            }
        }

        swap(&mut self.nrows, &mut self.ncols);
    }

    /// Shorthand call for transpose
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    /// matrix.t();
    ///
    /// assert_eq!(matrix.shape(), (100,2));
    /// ```
    pub fn t(&mut self) {
        self.transpose()
    }

    /// Transpose a matrix and return a copy
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,100));
    /// let result = matrix.transpose_copy();
    ///
    /// assert_eq!(result.shape(), (100,2));
    /// ```
    pub fn transpose_copy(&self) -> Self {
        let mut res = self.clone();
        res.transpose();
        res
    }
}

/// Implementations for predicates
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Counts all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0f32, (2,4));
    ///
    /// assert_eq!(matrix.count_where(|&e| e == 2.0), 8);
    /// ```
    pub fn count_where<F>(&'a self, pred: F) -> usize
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    /// Sums all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.sum_where(|&e| e == 2.0), 16.0);
    /// ```
    pub fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data
            .par_iter()
            .filter(|&e| pred(e))
            .copied()
            .sum::<T>()
    }

    /// Setsall elements where predicate holds true.
    /// The new value is to be set inside the predicate as well
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 2.0);
    ///
    /// matrix.set_where(|e| {
    ///     if *e == 2.0 {
    ///         *e = 2.3;
    ///     }
    /// });
    ///
    /// assert_eq!(matrix.get(0,0).unwrap(), 2.3);
    /// ```
    pub fn set_where<P>(&mut self, mut pred: P)
    where
        P: FnMut(&mut T) + Sync + Send,
    {
        self.data.iter_mut().for_each(|e| pred(e));
    }

    /// Return whether or not a predicate holds at least once
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.any(|&e| e == 2.0), true);
    /// ```
    pub fn any<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().any(pred)
    }

    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::randomize_range(1.0, 4.0, (2,4));
    ///
    /// assert_eq!(matrix.all(|&e| e >= 1.0), true);
    /// ```
    pub fn all<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().all(pred)
    }

    /// Finds first index where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2f32, (2,4));
    ///
    /// assert_eq!(matrix.find(|&e| e >= 1f32), Some((0,0)));
    /// ```
    pub fn find<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn(&T) -> bool + Sync,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(self.one_to_2d_idx(idx));
        }

        None
    }

    /// Finds all indeces where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.find_all(|&e| e >= 3.0), None);
    /// ```
    pub fn find_all<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&T) -> bool + Sync,
    {
        let data: Vec<Shape> = self
            .data
            .par_iter()
            .enumerate()
            .filter_map(|(idx, elem)| {
                if pred(elem) {
                    Some(self.one_to_2d_idx(idx))
                } else {
                    None
                }
            })
            .collect();

        if data.is_empty() {
            None
        } else {
            Some(data)
        }
    }
}
