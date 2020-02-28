use ndarray::{
    NdFloat,
    Array2,
    s,
};

use sprs;


/// Creates CSR matrix from given diagonals
///
/// The created matrix represents diagonal-like sparse matrix (DIA), but in CSR data storage
/// because sprs crate does not provide DIA matrices currently.
///
pub fn diags<T>(diags: Array2<T>, offsets: &[isize], shape: sprs::Shape) -> sprs::CsMat<T>
    where T: NdFloat
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

    let mut mat = sprs::TriMat::<T>::new(shape);

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

        let diag = if offset < 0 {
            if rows >= cols {
                row_head()
            } else {
                row_tail()
            }
        } else {
            if rows >= cols {
                row_tail()
            } else {
                row_head()
            }
        };

        for l in 0..n {
            mat.add_triplet(l + i, l + j, diag[l]);
        }
    }

    mat.to_csr()
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
}
