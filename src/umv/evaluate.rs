use ndarray::{
    Dimension,
    Axis,
    Array,
    Array1,
    Array2,
    ArrayView1,
    ArrayView2,
    s,
    stack,
};

use crate::{
    Real,
    Result,
    ndarrayext::{from_2d, digitize},
    util::dim_from_vec
};

use super::{CubicSmoothingSpline, NdSpline};


impl<'a, T> NdSpline<'a, T>
    where
        T: Real
{
    /// Implements evaluating the spline on the given mesh of Xi-sites
    ///
    /// The internal method to avoid copying coeffs array
    pub(crate) fn evaluate_spline(
        order: usize,
        pieces: usize,
        breaks: ArrayView1<'_, T>,
        coeffs: ArrayView2<'_, T>,
        xi: ArrayView1<'a, T>) -> Array2<T>
    {
        let edges = {
            let mesh = breaks.slice(s![1..-1]);
            let one = Array1::<T>::ones((1, ));
            let left_bound = &one * T::neg_infinity();
            let right_bound = &one * T::infinity();

            stack![Axis(0), left_bound, mesh, right_bound]
        };

        let mut indices = digitize(&xi, &edges);

        // Go to local coordinates
        let xi = {
            let indexed_breaks = indices.mapv(|i| breaks[i]);
            &xi - &indexed_breaks
        };

        // Apply nested multiplication

        // Returns NxM array of coeffs values for given 1xM indices array
        // where N is ndim and M is the size of xi
        let get_indexed_coeffs = |inds: &Array1<usize>| {
            // Returns Nx1 2-d array of coeffs by given index
            let coeffs_by_index = |&index| {
                coeffs.slice(s![.., index]).insert_axis(Axis(1))
            };

            // Get the M-sized vector of coeffs values Nx1 arrays
            // for all dimensions for 1xM indices array
            let indexed_coeffs: Vec<_> = inds
                .iter()
                .map(coeffs_by_index)
                .collect();

            stack(Axis(1), &indexed_coeffs).unwrap()
        };

        // Vectorized computing the spline pieces (polynoms) on the given data sites
        let mut values = get_indexed_coeffs(&indices);

        for _ in 1..order {
            indices += pieces;
            values = values * &xi + get_indexed_coeffs(&indices);
        }

        values
    }
}


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where
        T: Real,
        D: Dimension
{
    pub(super) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Result<Array<T, D>> {
        let axis = self.axis.unwrap();
        let mut shape_tmp = self.y.shape().to_owned();
        shape_tmp[axis.0] = xi.len();

        let shape: D = dim_from_vec(self.y.ndim(), shape_tmp);

        let yi_2d = self.spline.as_ref().unwrap().evaluate(xi);
        let yi = from_2d(&yi_2d, shape, axis)?.to_owned();

        Ok(yi)
    }
}
