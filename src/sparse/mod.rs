//!  Module for defining sparse matrices.
//!
//! # What are sparse matrices
//!
//! Generally speaking, matrices with a lot of 0s
//!
//! # How are they represented
//!
//! Since storing large sparse matrices in memory is expensive
//!
//!
//! # What data structure does sukker use?
//!
//! For now, a hash map where the keys are indeces in the matrix
//! and tha value is the value at that 2d index
#![warn(missing_docs)]

mod helper;

use helper::*;
use num_traits::Float;
use rand::Rng;

use itertools::Itertools;
use std::fmt::Display;
use std::fs;
use std::{collections::HashMap, error::Error, marker::PhantomData, str::FromStr};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{at, LinAlgFloats, Matrix, MatrixElement, MatrixError, Operation, Shape};

/// SparseMatrixData represents the datatype used to store information
/// about non-zero values in a general matrix.
///
/// The keys are the index to the position in data,
/// while the value is the value to be stored inside the matrix
pub type SparseMatrixData<'a, T> = HashMap<Shape, T>;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
/// Represents a sparse matrix and its data
pub struct SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Vector containing all data
    pub data: SparseMatrixData<'a, T>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Error for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Send for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Sync for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

impl<'a, T> FromStr for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
{
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the input string and construct the matrix dynamically
        let data = s
            .trim()
            .lines()
            .skip(1)
            .map(|l| {
                let entry: Vec<&str> = l.split_whitespace().collect();

                let row = entry[0].parse::<usize>().unwrap();
                let col = entry[1].parse::<usize>().unwrap();
                let val = entry[2].parse::<T>().unwrap();

                ((row, col), val)
            })
            .collect::<SparseMatrixData<T>>();

        let dims = s
            .trim()
            .lines()
            .nth(0)
            .unwrap()
            .split_whitespace()
            .map(|e| e.parse::<usize>().unwrap())
            .collect::<Vec<usize>>();

        Ok(Self::new(data, (dims[0], dims[1])))
    }
}

impl<'a, T> Display for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let elem = match self.data.get(&(i, j)) {
                    Some(&val) => val,
                    None => T::zero(),
                };

                write!(f, "{elem} ");
            }
            writeln!(f);
        }
        writeln!(f, "\ndtype = {}", std::any::type_name::<T>())
    }
}

impl<'a, T> Default for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Returns a sparse 3x3 identity matrix
    fn default() -> Self {
        Self {
            data: HashMap::new(),
            nrows: 0,
            ncols: 0,
            _lifetime: PhantomData::default(),
        }
    }
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Constructs a new sparse matrix based on a hashmap
    /// containing the indices where value is not 0
    ///
    /// This function does not check whether or not the
    /// indices are valid and according to shape. Use `reshape`
    /// to fix this issue.
    ///
    /// Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use sukker::{smd, SparseMatrix, SparseMatrixData};
    ///
    /// // Here we can use the smd! macro
    /// // to easily be able to set up a new hashmap
    /// let indexes: SparseMatrixData<f64> = smd![
    ///     ( (0, 0), 2.0),
    ///     ( (0, 3), 4.0),
    ///     ( (4, 5), 6.0),
    ///     ( (2, 7), 8.0)
    /// ];
    ///
    /// let sparse = SparseMatrix::<f64>::new(indexes, (3,3));
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.get(4,5), None);
    /// assert_eq!(sparse.get(0,1), Some(0.0));
    /// ```
    pub fn new(data: SparseMatrixData<'a, T>, shape: Shape) -> Self {
        Self {
            data,
            nrows: shape.0,
            ncols: shape.1,
            _lifetime: PhantomData::default(),
        }
    }

    /// Inits an empty matrix based on shape
    pub fn init(nrows: usize, ncols: usize) -> Self {
        Self {
            data: HashMap::new(),
            nrows,
            ncols,
            _lifetime: PhantomData::default(),
        }
    }

    /// Returns a sparse eye matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.ncols, 3);
    /// assert_eq!(sparse.nrows, 3);
    /// ```
    pub fn eye(size: usize) -> Self {
        let data: SparseMatrixData<'a, T> = (0..size)
            .into_par_iter()
            .map(|i| ((i, i), T::one()))
            .collect();

        Self::new(data, (size, size))
    }

    /// Produces an eye with the same shape as another
    /// sparse matrix
    pub fn eye_like(matrix: &Self) -> Self {
        Self::eye(matrix.nrows)
    }

    /// Same as eye
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f64>::identity(3);
    ///
    /// assert_eq!(sparse.ncols, 3);
    /// assert_eq!(sparse.nrows, 3);
    /// ```
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Creates a matrix with only one values at random
    /// locations
    ///
    /// Same as `random_like` but with range from 1.0..=1.0
    pub fn ones(sparsity: f64, shape: Shape) -> Self {
        Self::randomize_range(T::one(), T::one(), sparsity, shape)
    }

    /// Reshapes a sparse matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::identity(3);
    ///
    /// sparse.reshape(5,5);
    ///
    /// assert_eq!(sparse.ncols, 5);
    /// assert_eq!(sparse.nrows, 5);
    /// ```
    pub fn reshape(&mut self, nrows: usize, ncols: usize) {
        self.nrows = nrows;
        self.ncols = ncols;
    }

    /// Creates a sparse matrix from a already existent
    /// dense one.
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::{SparseMatrix, Matrix};
    ///
    /// let dense = Matrix::<i32>::eye(4);
    ///
    /// let sparse = SparseMatrix::from_dense(dense);
    ///
    /// assert_eq!(sparse.get(0,0), Some(1));
    /// assert_eq!(sparse.get(1,0), Some(0));
    /// assert_eq!(sparse.shape(), (4,4));
    /// ```
    pub fn from_dense(matrix: Matrix<'a, T>) -> Self {
        let mut data: SparseMatrixData<'a, T> = HashMap::new();

        for i in 0..matrix.nrows {
            for j in 0..matrix.ncols {
                let val = matrix.get(i, j).unwrap();
                if val != T::zero() {
                    data.insert((i, j), val);
                }
            }
        }

        Self::new(data, matrix.shape())
    }

    /// Constructs a sparse matrix from 3 slices.
    /// One for the rows, one for the cols, and one for the value.
    /// A combination of values fromt the same index corresponds to
    /// an entry in the hasmap.
    ///
    /// Csc in numpy uses 3 lists of same size
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let rows = vec![0,1,2,3];
    /// let cols = vec![1,2,3,4];
    /// let vals= vec![0.0,1.3,0.05,4.53];
    ///
    /// let shape = (6,7);
    ///
    /// let sparse = SparseMatrix::from_slices(&rows, &cols, &vals, shape).unwrap();
    ///
    /// assert_eq!(sparse.shape(), (6,7));
    /// assert_eq!(sparse.at(1,2), 1.3);
    /// assert_eq!(sparse.at(0,1), 0.0);
    /// ```
    pub fn from_slices(
        rows: &[usize],
        cols: &[usize],
        vals: &[T],
        shape: Shape,
    ) -> Result<Self, MatrixError> {
        if rows.len() != cols.len() && cols.len() != vals.len() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let data: SparseMatrixData<T> = rows
            .iter()
            .zip(cols.iter().zip(vals.iter()))
            .map(|(&i, (&j, &val))| ((i, j), val))
            .collect();

        Ok(Self::new(data, shape))
    }

    /// Parses from file, but will return a default sparse matrix if nothing is given
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// // let m: SparseMatrix<f32> = Matrix::from_file("../../test.txt").unwrap();
    ///
    /// // m.print(4);
    /// ```
    pub fn from_file(path: &'static str) -> Result<Self, MatrixError> {
        let data =
            fs::read_to_string(path).map_err(|_| MatrixError::MatrixFileReadError(path).into())?;

        data.parse::<Self>()
            .map_err(|_| MatrixError::MatrixParseError.into())
    }

    /// Gets an element from the sparse matrix.
    ///
    /// Returns None if index is out of bounds.
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.get(0,0), Some(1));
    /// assert_eq!(sparse.get(1,0), Some(0));
    /// assert_eq!(sparse.get(4,0), None);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        let idx = at!(i, j, self.ncols);

        if idx >= self.size() {
            eprintln!("Error, index out of bounds. Not setting value");
            return None;
        }

        match self.data.get(&(i, j)) {
            None => Some(T::zero()),
            val => val.copied(),
        }
    }

    /// Gets the size of the sparse matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::randomize_range(1.0,2.0, 0.75, (4,4));
    ///
    /// assert_eq!(sparse.shape(), (4,4));
    /// assert_eq!(sparse.sparsity(), 0.75);
    /// assert_eq!(sparse.all(|(_, val)| val >= 1.0 && val <= 2.0), true);
    /// assert_eq!(sparse.size(), 16);
    /// ```
    pub fn randomize_range(start: T, end: T, sparsity: f64, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        // If we insert in a position that's already filled up,
        // we ahve to get a new one
        let mut matrix = Self::init(shape.0, shape.1);

        while matrix.sparsity() > sparsity {
            let value: T = rng.gen_range(start..=end);

            let row: usize = rng.gen_range(0..rows);
            let col: usize = rng.gen_range(0..cols);

            match matrix.data.get(&(row, col)) {
                Some(_) => {}
                None => matrix.set(value, (row, col)),
            }
        }

        matrix
    }

    /// Randomizes a sparse matrix with values between 0 and 1.
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::randomize(0.75, (4,4));
    ///
    /// assert_eq!(sparse.shape(), (4,4));
    /// assert_eq!(sparse.sparsity(), 0.75);
    /// assert_eq!(sparse.all(|(_, val)| val >= 0.0 && val <= 1.0), true);
    /// assert_eq!(sparse.size(), 16);
    /// ```
    pub fn randomize(sparcity: f64, shape: Shape) -> Self {
        Self::randomize_range(T::zero(), T::one(), sparcity, shape)
    }

    /// Randomizes a sparse matrix  to have same shape and sparcity as another one
    /// You can however set the range
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::randomize_range(1.0,2.0, 0.75, (4,4));
    ///
    /// let copy = SparseMatrix::randomize_range_like(2.0, 4.0, &sparse);
    ///
    /// assert_eq!(copy.shape(), (4,4));
    /// assert_eq!(copy.sparsity(), 0.75);
    /// assert_eq!(copy.all(|(_, val)| val >= 2.0 && val <= 4.0), true);
    /// assert_eq!(copy.size(), 16);
    /// ```
    pub fn randomize_range_like(start: T, end: T, matrix: &Self) -> Self {
        Self::randomize_range(start, end, matrix.sparsity(), matrix.shape())
    }

    /// Randomizes a sparse matrix  to have same shape and sparcity as another one
    /// The values here are set to be between 0 and 1, no matter the value range
    /// of the matrix whos shape is being copied.
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::randomize_range(2.0, 4.0, 0.75, (4,4));
    ///
    /// let copy = SparseMatrix::random_like(&sparse);
    ///
    /// assert_eq!(copy.shape(), (4,4));
    /// assert_eq!(copy.sparsity(), 0.75);
    /// assert_eq!(copy.all(|(_, val)| val >= 0.0 && val <= 1.0), true);
    /// assert_eq!(copy.size(), 16);
    /// ```
    pub fn random_like(matrix: &Self) -> Self {
        Self::randomize(matrix.sparsity(), matrix.shape())
    }

    /// Same as `get`, but will panic if indexes are out of bounds
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.at(0,0), 1);
    /// assert_eq!(sparse.at(1,0), 0);
    /// ```
    #[inline(always)]
    pub fn at(&self, i: usize, j: usize) -> T {
        match self.data.get(&(i, j)) {
            None => T::zero(),
            Some(val) => val.clone(),
        }
    }

    /// Sets an element
    ///
    /// If you're trying to insert a zero-value, this function
    /// does nothing
    ///
    /// Mutates or inserts a value based on indeces given
    pub fn set(&mut self, value: T, idx: Shape) {
        if value == T::zero() {
            eprintln!("You are trying to insert a 0 value.");
            return;
        }

        let i = at!(idx.0, idx.1, self.ncols);

        if i >= self.size() {
            eprintln!("Error, index out of bounds. Not setting value");
            return;
        }

        self.data
            .entry(idx)
            .and_modify(|val| *val = value)
            .or_insert(value);
    }

    /// A way of inserting with individual row and col
    pub fn insert(&mut self, i: usize, j: usize, value: T) {
        self.set(value, (i, j));
    }

    /// Prints out the sparse matrix data
    ///
    /// Only prints out the hashmap with a set amount of decimals
    pub fn print(&self, decimals: usize) {
        self.data
            .iter()
            .for_each(|((i, j), val)| println!("{i} {j}: {:.decimals$}", val));
    }

    /// Gets the size of the sparse matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(4);
    ///
    /// assert_eq!(sparse.size(), 16);
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.ncols * self.nrows
    }

    /// Get's amount of 0s in the matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(4);
    ///
    /// assert_eq!(sparse.get_zero_count(), 12);
    #[inline(always)]
    pub fn get_zero_count(&self) -> usize {
        self.size() - self.data.len()
    }

    /// Calcualtes sparcity for the given matrix
    /// Sparity is defined as the percantage of the matrix
    /// filled with 0 values
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(4);
    ///
    /// assert_eq!(sparse.sparsity(), 0.75);
    /// ```
    #[inline(always)]
    pub fn sparsity(&self) -> f64 {
        1.0 - self.data.par_iter().count() as f64 / self.size() as f64
    }

    /// Shape of the matrix outputted as a tuple
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// ```
    pub fn shape(&self) -> Shape {
        (self.nrows, self.ncols)
    }

    /// Transpose the matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<i32>::init(4,4);
    ///
    /// sparse.set(1, (2,0));
    /// sparse.set(2, (3,0));
    /// sparse.set(3, (0,1));
    /// sparse.set(4, (0,2));
    ///
    /// sparse.transpose();
    ///
    /// assert_eq!(sparse.at(0,2), 1);
    /// assert_eq!(sparse.at(0,3), 2);
    /// assert_eq!(sparse.at(1,0), 3);
    /// assert_eq!(sparse.at(2,0), 4);
    ///
    /// // Old value is now gone
    /// assert_eq!(sparse.get(3,0), Some(0));
    /// ```
    pub fn transpose(&mut self) {
        let mut new_data: SparseMatrixData<T> = HashMap::new();

        for (&(i, j), &val) in self.data.iter() {
            new_data.insert((j, i), val);
        }

        self.data = new_data;

        swap(&mut self.nrows, &mut self.ncols);
    }

    /// Shorthand for `transpose`
    pub fn t(&mut self) {
        self.transpose();
    }

    /// Tranpose the matrix into a new copy
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut mat = SparseMatrix::<i32>::init(4,4);
    ///
    /// mat.set(1, (2,0));
    /// mat.set(2, (3,0));
    /// mat.set(3, (0,1));
    /// mat.set(4, (0,2));
    ///
    /// let sparse = mat.transpose_new();
    ///
    /// assert_eq!(sparse.at(0,2), 1);
    /// assert_eq!(sparse.at(0,3), 2);
    /// assert_eq!(sparse.at(1,0), 3);
    /// assert_eq!(sparse.at(2,0), 4);
    ///
    /// assert_eq!(sparse.get(3,0), Some(0));
    /// ```
    pub fn transpose_new(&self) -> Self {
        let mut res = self.clone();
        res.transpose();
        res
    }

    /// Finds max element of a sparse matrix
    /// Will return 0 if matrix is empty
    pub fn max(&self) -> T {
        let elem = self
            .data
            .iter()
            .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap());

        return match elem {
            Some((_, &v)) => v,
            None => T::zero(),
        };
    }

    /// Finds minimum element of a sparse matrix
    /// Will return 0 if matrix is empty
    pub fn min(&self) -> T {
        let elem = self
            .data
            .iter()
            .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap());

        return match elem {
            Some((_, &v)) => v,
            None => T::zero(),
        };
    }

    /// Negates all items
    pub fn neg(&self) -> Self {
        let data = self
            .data
            .par_iter()
            .map(|((i, j), &e)| ((*i, *j), e.neg()))
            .collect::<SparseMatrixData<T>>();

        Self::new(data, self.shape())
    }

    /// Finds average value of a matrix
    ///
    /// Returns 0 if matrix is empty
    pub fn avg(&self) -> T {
        self.data.par_iter().map(|(_, &val)| val).sum::<T>()
            / self.size().to_string().parse::<T>().unwrap()
    }

    /// Same as `avg`
    pub fn mean(&self) -> T {
        self.avg()
    }

    /// Finds the median value of a matrix
    ///
    /// If the matrix is empty, 0 is returned
    pub fn median(&self) -> T {
        if self.size() == 0 {
            return T::zero();
        }

        if self.size() == 1 {
            return self.at(0, 0);
        }

        // If more than half the values are 0 and we only have
        // values > 0, 0 is returned
        if self.min() >= T::zero() && self.sparsity() >= 0.5 {
            return T::zero();
        }

        let sorted_values: Vec<T> = self
            .data
            .values()
            .copied()
            .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
            .collect::<Vec<T>>();

        match self.data.len() % 2 {
            0 => {
                let half: usize = self.data.len() / 2;

                sorted_values
                    .iter()
                    .skip(half - 1)
                    .take(2)
                    .copied()
                    .sum::<T>()
                    / (T::one() + T::one())
            }
            1 => {
                let half: usize = self.data.len() / 2;

                sorted_values.iter().nth(half).unwrap().to_owned()
            }
            _ => unreachable!(),
        }
    }
}

/// Linear algebra on sparse matrices
impl<'a, T> LinAlgFloats<'a, T> for SparseMatrix<'a, T>
where
    T: MatrixElement + Float,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn ln(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.ln())).collect();

        Self::new(data, self.shape())
    }

    fn log(&self, base: T) -> Self {
        let data = self
            .data
            .iter()
            .map(|(&idx, &e)| (idx, e.log(base)))
            .collect();

        Self::new(data, self.shape())
    }

    fn sin(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.sin())).collect();

        Self::new(data, self.shape())
    }

    fn cos(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.cos())).collect();

        Self::new(data, self.shape())
    }

    fn tan(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.tan())).collect();

        Self::new(data, self.shape())
    }

    fn sqrt(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.sqrt())).collect();

        Self::new(data, self.shape())
    }

    fn sinh(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.sinh())).collect();

        Self::new(data, self.shape())
    }

    fn cosh(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.cosh())).collect();

        Self::new(data, self.shape())
    }

    fn tanh(&self) -> Self {
        let data = self.data.iter().map(|(&idx, &e)| (idx, e.tanh())).collect();

        Self::new(data, self.shape())
    }

    fn get_eigenvalues(&self) -> Option<Vec<T>> {
        unimplemented!()
    }

    fn get_eigenvectors(&self) -> Option<Vec<T>> {
        unimplemented!()
    }
}

/// Operations on sparse matrices
impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Adds two sparse matrices together
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.add(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::ADD)
    }

    /// Subtracts two sparse matrices
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.sub(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::SUB)
    }
    /// Multiplies two sparse matrices together
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.mul(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::MUL)
    }

    /// Same as `mul`. This kind of matrix multiplication is called
    /// a dot product
    pub fn dot(&self, other: &Self) -> Result<Self, MatrixError> {
        self.mul(other)
    }

    /// Divides two sparse matrices
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.div(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::DIV)
    }

    // =============================================================
    //
    //    Matrix operations modifying the lhs
    //
    // =============================================================

    /// Adds rhs matrix on to lhs matrix.
    /// All elements from rhs gets inserted into lhs
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// sparse1.add_self(&sparse2);
    ///
    /// assert_eq!(sparse1.shape(), (3,3));
    /// assert_eq!(sparse1.get(0,0).unwrap(), 2);
    /// ```
    pub fn add_self(&mut self, other: &Self) {
        Self::sparse_helper_self(self, other, Operation::ADD);
    }

    /// Subs rhs matrix on to lhs matrix.
    /// All elements from rhs gets inserted into lhs
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// sparse1.sub_self(&sparse2);
    ///
    /// assert_eq!(sparse1.shape(), (3,3));
    /// assert_eq!(sparse1.get(0,0).unwrap(), 0);
    /// ```
    pub fn sub_self(&mut self, other: &Self) {
        Self::sparse_helper_self(self, other, Operation::SUB);
    }

    /// Multiplies  rhs matrix on to lhs matrix.
    /// All elements from rhs gets inserted into lhs
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// sparse1.mul_self(&sparse2);
    ///
    /// assert_eq!(sparse1.shape(), (3,3));
    /// assert_eq!(sparse1.get(0,0).unwrap(), 1);
    /// ```
    pub fn mul_self(&mut self, other: &Self) {
        Self::sparse_helper_self(self, other, Operation::MUL);
    }

    /// Divides rhs matrix on to lhs matrix.
    /// All elements from rhs gets inserted into lhs
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// sparse1.div_self(&sparse2);
    ///
    /// assert_eq!(sparse1.shape(), (3,3));
    /// assert_eq!(sparse1.get(0,0).unwrap(), 1);
    /// ```
    pub fn div_self(&mut self, other: &Self) {
        Self::sparse_helper_self(self, other, Operation::DIV);
    }

    // =============================================================
    //
    //    Matrix operations  with a value
    //
    // =============================================================

    /// Adds value to all non zero values in the matrix
    /// and return a new matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    /// let val: f32 = 4.5;
    ///
    /// let res = sparse.add_val(val);
    ///
    /// assert_eq!(res.get(0,0).unwrap(), 5.5);
    /// ```
    pub fn add_val(&self, value: T) -> Self {
        Self::sparse_helper_val(self, value, Operation::ADD)
    }

    /// Subs value to all non zero values in the matrix
    /// and return a new matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    /// let val: f32 = 4.5;
    ///
    /// let res = sparse.sub_val(val);
    ///
    /// assert_eq!(res.get(0,0).unwrap(), -3.5);
    /// ```
    pub fn sub_val(&self, value: T) -> Self {
        Self::sparse_helper_val(self, value, Operation::SUB)
    }

    /// Multiplies value to all non zero values in the matrix
    /// and return a new matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    /// let val: f32 = 4.5;
    ///
    /// let res = sparse.mul_val(val);
    ///
    /// assert_eq!(res.get(0,0).unwrap(), 4.5);
    /// ```
    pub fn mul_val(&self, value: T) -> Self {
        Self::sparse_helper_val(self, value, Operation::MUL)
    }

    /// Divides value to all non zero values in the matrix
    /// and return a new matrix.
    ///
    /// Will panic if you choose to divide by zero
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    /// let val: f32 = 4.0;
    ///
    /// let res = sparse.div_val(val);
    ///
    /// assert_eq!(res.get(0,0).unwrap(), 0.25);
    /// ```
    pub fn div_val(&self, value: T) -> Self {
        Self::sparse_helper_val(self, value, Operation::DIV)
    }

    // =============================================================
    //
    //    Matrix operations modyfing lhs  with a value
    //
    // =============================================================

    /// Adds value to all non zero elements in matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::eye(3);
    /// let val = 10.0;
    ///
    /// sparse.add_val_self(val);
    ///
    /// assert_eq!(sparse.get(0,0).unwrap(), 11.0);
    /// ```
    pub fn add_val_self(&mut self, value: T) {
        Self::sparse_helper_self_val(self, value, Operation::ADD)
    }

    /// Subtracts value to all non zero elements in matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::eye(3);
    /// let val = 10.0;
    ///
    /// sparse.sub_val_self(val);
    ///
    /// assert_eq!(sparse.get(0,0).unwrap(), -9.0);
    /// ```
    pub fn sub_val_self(&mut self, value: T) {
        Self::sparse_helper_self_val(self, value, Operation::SUB)
    }

    /// Multiplies value to all non zero elements in matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::eye(3);
    /// let val = 10.0;
    ///
    /// sparse.mul_val_self(val);
    ///
    /// assert_eq!(sparse.get(0,0).unwrap(), 10.0);
    /// ```
    pub fn mul_val_self(&mut self, value: T) {
        Self::sparse_helper_self_val(self, value, Operation::MUL)
    }

    /// Divides all non zero elemnts in matrix by value in-place
    ///
    /// Will panic if you choose to divide by zero
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::eye(3);
    /// let val = 10.0;
    ///
    /// sparse.div_val_self(val);
    ///
    /// assert_eq!(sparse.get(0,0).unwrap(), 0.1);
    /// ```
    pub fn div_val_self(&mut self, value: T) {
        Self::sparse_helper_self_val(self, value, Operation::DIV)
    }

    /// Sparse matrix multiplication
    ///
    /// For two n x n matrices, we use this algorithm:
    /// https://theory.stanford.edu/~virgi/cs367/papers/sparsemult.pdf
    ///
    /// Else, we use this:
    /// link..
    ///
    /// In this example we have these two matrices:
    ///
    /// A:
    ///
    /// 0.0 2.0 0.0
    /// 4.0 6.0 0.0
    /// 0.0 0.0 8.0
    ///
    /// B:
    ///
    /// 2.0 0.0 0.0
    /// 4.0 8.0 0.0
    /// 8.0 6.0 0.0
    ///
    /// 0.0 24.0 0
    /// 8.0 72.0 0
    /// 0.0 0.0  48.0
    ///
    /// Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use sukker::{SparseMatrix, SparseMatrixData};
    ///
    /// let mut indexes: SparseMatrixData<f64> = HashMap::new();
    ///
    /// indexes.insert((0, 0), 2.0);
    /// indexes.insert((0, 1), 2.0);
    /// indexes.insert((1, 0), 2.0);
    /// indexes.insert((1, 1), 2.0);
    ///
    /// let sparse = SparseMatrix::<f64>::new(indexes, (2, 2));
    ///
    /// let mut indexes2: SparseMatrixData<f64> = HashMap::new();
    ///
    /// indexes2.insert((0, 0), 2.0);
    /// indexes2.insert((0, 1), 2.0);
    /// indexes2.insert((1, 0), 2.0);
    /// indexes2.insert((1, 1), 2.0);
    ///
    /// let sparse2 = SparseMatrix::<f64>::new(indexes2, (2, 2));
    ///
    /// let res = sparse.matmul_sparse(&sparse2).unwrap();
    ///
    /// assert_eq!(res.at(0, 0), 8.0);
    /// assert_eq!(res.at(0, 1), 8.0);
    /// assert_eq!(res.at(1, 0), 8.0);
    /// assert_eq!(res.at(1, 1), 8.0);
    /// ```
    pub fn matmul_sparse(&self, other: &Self) -> Result<Self, MatrixError> {
        if self.ncols != other.nrows {
            return Err(MatrixError::MatrixMultiplicationDimensionMismatchError.into());
        }

        if self.shape() == other.shape() {
            return Ok(self.matmul_sparse_nn(other));
        }

        Ok(self.matmul_sparse_mnnp(other))
    }
}

/// Predicates for sparse matrices
impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.all(|(idx, val)| val >= 0), true);
    /// ```
    pub fn all<F>(&self, pred: F) -> bool
    where
        F: Fn((Shape, T)) -> bool + Sync + Send,
    {
        self.data.clone().into_par_iter().all(pred)
    }

    /// Returns whether or not predicate holds for any
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.any(|(_, val)| val == 1), true);
    /// ```
    pub fn any<F>(&self, pred: F) -> bool
    where
        F: Fn((Shape, T)) -> bool + Sync + Send,
    {
        self.data.clone().into_par_iter().any(pred)
    }

    /// Counts all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.count_where(|(_, &val)| val == 1), 3);
    /// ```
    pub fn count_where<F>(&'a self, pred: F) -> usize
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    /// Sums all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    ///
    /// assert_eq!(sparse.sum_where(|(&(i, j), &val)| val == 1.0 && i > 0), 2.0);
    /// ```
    pub fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        let mut res = T::zero();
        for (idx, elem) in self.data.iter() {
            if pred((idx, elem)) {
                res += elem
            }
        }

        res
    }

    /// Sets all elements where predicate holds true.
    /// The new value is to be set inside the predicate as well
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    pub fn set_where<F>(&mut self, mut pred: F)
    where
        F: FnMut((&Shape, &mut T)) + Sync + Send,
    {
        self.data.iter_mut().for_each(|e| pred(e));
    }

    /// Finds value of first occurance where predicate holds true
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    pub fn find<F>(&self, pred: F) -> Option<T>
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        for entry in &self.data {
            if pred(entry) {
                return Some(*entry.1);
            }
        }

        None
    }

    /// Finds all values where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    fn find_all<F>(&self, pred: F) -> Option<Vec<T>>
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        let mut idxs: Vec<T> = Vec::new();
        for entry in &self.data {
            if pred(entry) {
                idxs.push(*entry.1);
            }
        }

        if !idxs.is_empty() {
            Some(idxs)
        } else {
            None
        }
    }

    /// Finds indices of first occurance where predicate holds true
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    pub fn position<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        for entry in &self.data {
            if pred(entry) {
                return Some(*entry.0);
            }
        }

        None
    }

    /// Finds all positions  where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// ```
    fn positions<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn((&Shape, &T)) -> bool + Sync,
    {
        let mut idxs: Vec<Shape> = Vec::new();
        for entry in &self.data {
            if pred(entry) {
                idxs.push(*entry.0);
            }
        }

        if !idxs.is_empty() {
            Some(idxs)
        } else {
            None
        }
    }
}
