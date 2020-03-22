use std::iter::FromIterator;

use ndarray::{Array1, Array2, Axis, s};

use sprs::{TriMat, CsMat, Shape};
use sprs_ldl::LdlNumeric;

use crate::Real;


/// Creates CSR matrix from given diagonals
///
/// The created matrix represents diagonal-like sparse matrix (DIA), but in CSR data storage
/// because sprs crate does not provide DIA matrices currently.
///
pub fn diags<T>(diags: Array2<T>, offsets: &[isize], shape: Shape) -> CsMat<T>
    where
        T: Real
{
    let (rows, cols) = shape;

    let numel_and_indices = |offset: isize| {
        let mut i: usize = 0;
        let mut j: usize = 0;

        if offset < 0 {
            i = offset.abs() as usize;
        } else {
            j = offset as usize;
        }

        ((rows - i).min(cols - j), i, j)
    };

    let mut mat = TriMat::<T>::new(shape);

    for (k, &offset) in offsets.iter().enumerate() {
        let (n, i, j) = numel_and_indices(offset);

        // When rows == cols or rows > cols, the function takes elements of the
        // super-diagonal from the lower part of the corresponding diag array, and
        // elements of the sub-diagonal from the upper part of the corresponding diag array.
        //
        // When rows < cols, the function does the opposite, taking elements of the
        // super-diagonal from the upper part of the corresponding diag array, and
        // elements of the sub-diagonal from the lower part of the corresponding diag array.
        let diag_row = diags.row(k);

        let row_head = || diag_row.slice(s![..n]);
        let row_tail = || diag_row.slice(s![-(n as isize)..]);

        let diag = if offset >= 0 {
            if rows >= cols {
                row_tail()
            } else {
                row_head()
            }
        } else {
            if rows >= cols {
                row_head()
            } else {
                row_tail()
            }
        };

        for l in 0..n {
            mat.add_triplet(l + i, l + j, diag[l]);
        }
    }

    mat.to_csr()
}


/// Returns values on k-diagonal for given sparse matrix
///
///
pub fn diagonal<T>(m: &CsMat<T>, k: isize) -> Array1<T>
    where
        T: Real
{
    let (rows, cols) = m.shape();

    if m.is_csr() {
        diagonal_csr(k, (rows, cols), m.indptr(), m.indices(), m.data())
    } else {
        diagonal_csr(-k, (cols, rows), m.indptr(), m.indices(), m.data())
    }
}


fn diagonal_csr<T>(k: isize,
                shape: Shape,
                indptr: &[usize],
                indices: &[usize],
                data: &[T]) -> Array1<T>
    where
        T: Real
{
    let (rows, cols) = shape;

    if k <= -(rows as isize) || k >= cols as isize {
        panic!(format!("k ({}) exceeds matrix dimensions {:?}", k, shape));
    }

    let first_row = if k >= 0 { 0 } else { (-k) as usize };
    let first_col = if k >= 0 { k as usize } else { 0 };

    let diag_size = (rows - first_row).min(cols - first_col) as usize;
    let mut diag = Array1::<T>::zeros((diag_size, ));

    for i in 0..diag_size {
        let row = first_row + i;
        let col = first_col + i;
        let row_begin = indptr[row];
        let row_end = indptr[row + 1];

        let mut diag_value = T::zero();

        for j in row_begin..row_end {
            if indices[j] == col {
                diag_value += data[j];
            }
        }

        diag[i] = diag_value;
    }

    diag
}


/// Solves linear system Ax = b for symmetric CSR matrix A and dense vector(s) b
///
/// A: ref to CSR symmetric sparse matrix
/// b: MxN stack of b-vectors where M is equal to A rows/cols and N is the data dimensional
///
pub fn solve<T>(a: &CsMat<T>, b: &Array2<T>) -> Array2<T>
    where
        T: Real
{
    let mut x = Array2::<T>::zeros(b.raw_dim());

    let b_col_iter = b.axis_iter(Axis(1));
    let x_col_iter = x.axis_iter_mut(Axis(1));

    for (b_col, mut x_col) in b_col_iter.zip(x_col_iter) {
        let xi = {
            let x_vec = if b_col.len() > 1 {
                let ldl = LdlNumeric::new(a.view()).unwrap();
                let b_vec = Vec::from_iter(b_col.iter().cloned());
                ldl.solve(&b_vec)
            } else {
                // In this corner case we have 1x1 sparse matrix and `b` vector with 1 element
                vec![b_col[0] / a.to_dense()[[0, 0]]]
            };

            Array1::from(x_vec)
        };

        x_col.zip_mut_with(&xi, |v1, v2| *v1 = *v2);
    }

    x
}


#[cfg(test)]
mod tests {
    use ndarray::array;
    use sprs::Shape;

    use crate::sprsext;

    #[test]
    fn test_diags_1() {
        /*
            4     8     0
            1     5     9
            0     2     6
        */

        let diags = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ];

        let shape = (3, 3);

        let mat = sprsext::diags(diags, &[-1, 0, 1], shape);

        let mat_expected = sprs::TriMat::<f64>::from_triplets(
            shape,
            vec![0, 1, 0, 1, 2, 1, 2],
            vec![0, 0, 1, 1, 1, 2, 2],
            vec![4., 1., 8., 5., 2., 9., 6.],
        ).to_csr();

        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_diags_2() {
        /*
            4     7     0     0     0
            2     5     8     0     0
            0     3     6     9     0
        */

        let diags = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ];

        let shape = (3, 5);

        let mat = sprsext::diags(diags, &[-1, 0, 1], shape);

        let mat_expected = sprs::TriMat::<f64>::from_triplets(
            shape,
            vec![0, 1, 0, 1, 2, 1, 2, 2],
            vec![0, 0, 1, 1, 1, 2, 2, 3],
            vec![4., 2., 7., 5., 3., 8., 6., 9.],
        ).to_csr();

        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_diags_3() {
        /*
            4     8     0
            1     5     9
            0     2     6
            0     0     3
            0     0     0
        */

        let diags = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ];

        let shape: Shape = (5, 3);

        let mat = sprsext::diags(diags, &[-1, 0, 1], shape);

        let mat_expected = sprs::TriMat::<f64>::from_triplets(
            shape,
            vec![0, 1, 0, 1, 2, 1, 2, 3],
            vec![0, 0, 1, 1, 1, 2, 2, 2],
            vec![4., 1., 8., 5., 2., 9., 6., 3.],
        ).to_csr();

        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_diags_4() {
        /*
            7     0     0
            4     8     0
            1     5     9
            0     2     6
            0     0     3
        */

        let diags = array![
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.],
        ];

        let shape: Shape = (5, 3);

        let mat = sprsext::diags(diags, &[-2, -1, 0], shape);

        let mat_expected = sprs::TriMat::<f64>::from_triplets(
            shape,
            vec![0, 1, 2, 1, 2, 3, 2, 3, 4],
            vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
            vec![7., 4., 1., 8., 5., 2., 9., 6., 3.],
        ).to_csr();

        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_diags_5() {
        /*
             1     0     0
             0     2     0
             0     0     3
        */

        let shape: Shape = (3, 3);

        let mat = sprsext::diags(array![[1., 2., 3.]], &[0], shape);

        let mat_expected = sprs::TriMat::<f64>::from_triplets(
            shape,
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1., 2., 3.],
        ).to_csr();

        assert_eq!(mat, mat_expected);
    }

    #[test]
    fn test_diagonal_1() {
        let k = 0;
        let m_csr = sprsext::diags(array![[1., 2., 3.]], &[k], (3, 3));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2., 3.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2., 3.]);
    }

    #[test]
    fn test_diagonal_2() {
        let k = -1;
        let m_csr = sprsext::diags(array![[1., 2.]], &[k], (3, 3));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2.]);
    }

    #[test]
    fn test_diagonal_3() {
        let k = 1;
        let m_csr = sprsext::diags(array![[1., 2.]], &[k], (3, 3));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2.]);
    }

    #[test]
    fn test_diagonal_4() {
        let k = -2;
        let m_csr = sprsext::diags(array![[1., 2., 3.]], &[k], (5, 3));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2., 3.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2., 3.]);
    }

    #[test]
    fn test_diagonal_5() {
        let k = 1;
        let m_csr = sprsext::diags(array![[1., 2., 3.]], &[k], (3, 5));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2., 3.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2., 3.]);
    }

    #[test]
    fn test_diagonal_6() {
        let k = -1;
        let m_csr = sprsext::diags(array![[1., 2.]], &[k], (3, 5));
        let m_csc = m_csr.to_csc();

        assert_eq!(sprsext::diagonal(&m_csr, k), array![1., 2.]);
        assert_eq!(sprsext::diagonal(&m_csc, k), array![1., 2.]);
    }
}
