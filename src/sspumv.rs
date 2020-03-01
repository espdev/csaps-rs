use std::iter::FromIterator;

use ndarray::{
    NdFloat,
    Dimension,
    Array,
    Array1,
    ArrayView1,
    Axis,
    s,
    stack,
};

use sprs::binop::scalar_mul_mat as muls;
use sprs::linalg::trisolve::lsolve_csr_dense_rhs as solve;

use crate::{
    CubicSmoothingSpline,
    Result,
    validatedata,
    ndarrayext,
    sprsext,
};
use std::error::Error;


impl<'a, T, D> CubicSmoothingSpline<'a, T, D>
    where T: NdFloat + Default, D: Dimension
{
    pub(crate) fn make_spline(&mut self) -> Result<()> {
        let weights_default = Array1::ones(self.x.raw_dim());
        let weights = self.weights
            .map(|v| v.reborrow()) // without it we will get an error: "[E0597] `weights_default` does not live long enough"
            .unwrap_or(weights_default.view());

        let dx = ndarrayext::diff(self.x.view(), None);

        validatedata::validate_sites_increase(&dx)?;

        let axis = self.axis.unwrap_or(Axis(self.y.ndim() - 1));

        let y = ndarrayext::to_2d(self.y.view(), axis)?;
        let dydx = ndarrayext::diff(y.view(), Some(Axis(1))) / &dx;

        let pcount = self.x.len();

        // The corner case for Nx2 data (2 data points)
        if pcount == 2 {
            drop(dx);
            let yi = y.slice(s![.., 0]).insert_axis(Axis(1));

            self.coeffs = Some(stack![Axis(1), dydx, yi]);
            self.smooth = Some(T::one());
            self.order = Some(2);
            self.pieces = Some(1);
            self.is_valid = true;

            return Ok(())
        }

        // General smoothing spline computing for NxM data (2 and more data points)
        let ones = |n| Array1::<T>::ones((n, ));

        let qt = {
            let odx = ones(pcount - 1) / &dx;
            let odx_head = odx.slice(s![..-1]).insert_axis(Axis(0)).into_owned();
            let odx_tail = odx.slice(s![1..]).insert_axis(Axis(0)).into_owned();
            let odx_body = -(&odx_tail + &odx_head);
            let diags_qt = stack![Axis(0), odx_head, odx_body, odx_tail];

            sprsext::diags(diags_qt, &[0, 1, 2], (pcount - 2, pcount))
        };

        let qtwq = {
            let diags_sqrw = (ones(pcount) / weights.mapv(T::sqrt)).insert_axis(Axis(0));
            let sqrw = sprsext::diags(diags_sqrw, &[0], (pcount, pcount));
            let qtw = &qt * &sqrw;
            let qtw_t = qtw.transpose_view();
            drop(qt);

            &qtw * &qtw_t
        };

        let r = {
            let dx_head = dx.slice(s![1..]).insert_axis(Axis(0)).into_owned();
            let dx_tail = dx.slice(s![..-1]).insert_axis(Axis(0)).into_owned();
            let dx_body = (&dx_head + &dx_tail) * T::from(2.0).unwrap();
            let diags_r = stack![Axis(0), dx_head, dx_body, dx_tail];

            sprsext::diags(diags_r, &[-1, 0, 1], (pcount - 2, pcount - 2))
        };

        let one = T::one();
        let six = T::from(6.0).unwrap();

        let auto_smooth = || {
            let trace = |m| { sprsext::diagonal(m, 0).sum() };
            one / (one + trace(&r) / (six * trace(&qtwq)))
        };

        let p = self.smooth.unwrap_or_else(auto_smooth);

        // # Solve linear system Ax = b for the 2nd derivatives
        let u = {
            let a = {
                let a1 = muls(&qtwq, six * (one - p));
                let a2 = muls(&r, p);
                drop(qtwq);
                drop(r);

                &a1 + &a2
            };

            let dydx_d = ndarrayext::diff(&dydx, Some(Axis(1))).t().to_owned();
            drop(dydx);

            // FIXME: currently, it does not work for Y ndim > 1
            if self.ndim > 1 {
                unimplemented!();
            }

            let mut b = Vec::from_iter(dydx_d.iter().cloned());

            if let Err(error) = solve(a.view(), &mut b) {
                return Err(format!(
                    "Cannot solve linear system for the 2nd derivatives. Error: {}", error.description())
                )
            };

            Array1::<T>::from(b)
        };

        // Compute and stack spline coefficients

        let w = {
            let diags_w = (ones(pcount) / &weights).insert_axis(Axis(0));
            sprsext::diags(diags_w, &[0], (pcount, pcount))
        };

        unimplemented!();

        self.smooth = Some(p);

        Ok(())
    }

    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        unimplemented!();
    }
}
