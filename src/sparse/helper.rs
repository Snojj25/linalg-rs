//! Internal helpers

use std::{error::Error, str::FromStr};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{MatrixElement, MatrixError, Operation, SparseMatrix, SparseMatrixData};

pub fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    // Helper for add, sub, mul and div on SparseMatrix - SparseMatrix operation
    #[doc(hidden)]
    pub fn sparse_helper(&self, other: &Self, op: Operation) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let mut result_mat = Self::init(self.nrows, self.ncols);

        for (&idx, &val) in self.data.iter() {
            result_mat.set(val, idx);
        }

        for (&idx, &val) in other.data.iter() {
            match result_mat.data.get_mut(&idx) {
                Some(value) => match op {
                    Operation::ADD => *value += val,
                    Operation::SUB => *value += val,
                    Operation::MUL => *value += val,
                    Operation::DIV => *value += val,
                },
                None => result_mat.set(val, idx),
            };
        }

        Ok(result_mat)
    }

    #[doc(hidden)]
    pub fn sparse_helper_self(&mut self, other: &Self, op: Operation) {
        // TODO: Mismatch in dimensions might not be an issue? Find out
        if self.shape() != other.shape() {
            eprintln!("Oops, mismatch in dims");
            return;
        }

        for (&idx, &val) in other.data.iter() {
            match self.data.get_mut(&idx) {
                Some(value) => match op {
                    Operation::ADD => *value += val,
                    Operation::SUB => *value -= val,
                    Operation::MUL => *value *= val,
                    Operation::DIV => *value /= val,
                },
                None => self.set(val, idx),
            };
        }
    }

    #[doc(hidden)]
    pub fn sparse_helper_val(&self, value: T, op: Operation) -> Self {
        let mut result_mat = Self::init(self.nrows, self.ncols);

        for (&idx, &old_value) in self.data.iter() {
            let new_value = match op {
                Operation::ADD => old_value + value,
                Operation::SUB => old_value - value,
                Operation::MUL => old_value * value,
                Operation::DIV => old_value / value,
            };

            result_mat.set(new_value, idx);
        }

        result_mat
    }

    #[doc(hidden)]
    pub fn sparse_helper_self_val(&mut self, val: T, op: Operation) {
        for (_, value) in self.data.iter_mut() {
            match op {
                Operation::ADD => *value += val,
                Operation::SUB => *value -= val,
                Operation::MUL => *value *= val,
                Operation::DIV => *value /= val,
            }
        }
    }

    // =============================================================
    //     Sparse Matrix Mulitplication helpers
    // =============================================================

    // For now these algorithms are the same since we're using
    // hashmaps

    // For nn x nn
    #[doc(hidden)]
    pub fn matmul_sparse_nn(&self, other: &Self) -> Self {
        // For now, more or less same as mn np
        let N = self.nrows;

        let data: SparseMatrixData<T> = (0..N)
            .flat_map(|i| (0..N).map(move |j| (i, j)))
            .collect::<Vec<(usize, usize)>>()
            .into_par_iter()
            .filter_map(|(i, j)| {
                if self.at(i, j) == T::zero() {
                    return None;
                }

                let result = (0..N)
                    .into_par_iter()
                    .map(|k| self.at(i, j) * other.at(j, k))
                    .sum();

                Some(((i, j), result))
            })
            .collect();

        Self::new(data, (N, N))
    }

    // mn x np
    #[doc(hidden)]
    pub fn matmul_sparse_mnnp(&self, other: &Self) -> Self {
        let x = self.nrows;
        let y = self.ncols;
        let z = other.ncols;

        let data: SparseMatrixData<T> = (0..x)
            .flat_map(|i| (0..y).map(move |j| (i, j)))
            .collect::<Vec<(usize, usize)>>()
            .into_par_iter()
            .filter_map(|(i, j)| {
                if self.at(i, j) == T::zero() {
                    return None;
                }

                let result = (0..z)
                    .into_par_iter()
                    .map(|k| self.at(i, j) * other.at(j, k))
                    .sum();

                Some(((i, j), result))
            })
            .collect();

        Self::new(data, (x, z))
    }
}
