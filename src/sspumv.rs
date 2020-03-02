use ndarray::{NdFloat, Dimension, Array, Array1, ArrayView1, Axis, s, stack, Array2};

use sprs::binop::scalar_mul_mat as muls;

use crate::{
    CubicSmoothingSpline,
    Result,
    validatedata,
    ndarrayext,
    sprsext,
};


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

        let ndim = y.shape()[0];
        let pcount = self.x.len();

        // The corner case for Nx2 data (2 data points)
        if pcount == 2 {
            drop(dx);
            let yi = y.slice(s![.., 0]).insert_axis(Axis(1));

            self.smooth = Some(T::one());
            self.coeffs = Some(stack![Axis(1), dydx, yi]);
            self.ndim = Some(ndim);
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
            drop(odx);
            let odx_body = -(&odx_tail + &odx_head);
            let diags_qt = stack![Axis(0), odx_head, odx_body, odx_tail];

            sprsext::diags(diags_qt, &[0, 1, 2], (pcount - 2, pcount))
        };

        let qtwq = {
            let diags_sqrw = (ones(pcount) / weights.mapv(T::sqrt)).insert_axis(Axis(0));
            let sqrw = sprsext::diags(diags_sqrw, &[0], (pcount, pcount));
            let qtw = &qt * &sqrw;
            drop(sqrw);
            drop(qt);
            let qtw_t = qtw.transpose_view();

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
        let p1 = six * (one - p);

        // # Solve linear system Ax = b for the 2nd derivatives
        let u = {
            let a = {
                let a1 = muls(&qtwq, p1);
                let a2 = muls(&r, p);
                drop(qtwq);
                drop(r);

                &a1 + &a2
            };

            let b = ndarrayext::diff(&dydx, Some(Axis(1))).t().to_owned();
            drop(dydx);

            sprsext::solve(&a, &b)?
        };

        // Compute and stack spline coefficients

        let vpad = |a: &Array2<T>| -> Array2<T> {
            let p = Array2::<T>::zeros((1, a.shape()[1]));
            stack(Axis(0), &[p.view(), a.view(), p.view()]).unwrap()
        };

        let dx = dx.insert_axis(Axis(1));

        let (d1, d2) = {
            let u_pad = vpad(&u);
            let d1 = ndarrayext::diff(&u_pad, Some(Axis(0))) / &dx;
            let d1_pad = vpad(&d1);
            let d2 = ndarrayext::diff(&d1_pad, Some(Axis(0)));

            (d1, d2)
        };

        let wd2 = {
            let diags_w = (ones(pcount) / &weights).insert_axis(Axis(0));
            let w = sprsext::diags(diags_w, &[0], (pcount, pcount));

            let wd2 = &muls(&w, p1) * &d2;
            drop(d1);
            drop(d2);
            wd2
        };

        let yi = &y.t() - &wd2;

        let c3 = vpad(&(u * p));
        let c3_head = c3.slice(s![..-1, ..]);
        let c3_tail = c3.slice(s![1.., ..]);

        let c2 = {
            let dyi = ndarrayext::diff(&yi, Some(Axis(0)));
            let two = T::from(2.0).unwrap();
            let c32sum = &c3_head * two + c3_tail;

            &dyi / &dx - &(&c32sum * &dx)
        };

        let order = 4;

        let (coeffs, pieces) = {
            let c3ddx = ndarrayext::diff(&c3, Some(Axis(0))) / &dx;
            let three = T::from(2.0).unwrap();
            let c3head3 = &c3_head * three;
            let yi_head = yi.slice(s![..-1, ..]);

            let coeffs = stack![Axis(0), c3ddx, c3head3, c2, yi_head].t().to_owned();
            let pieces = coeffs.shape()[1] / order;

            (coeffs, pieces)
        };

        self.smooth = Some(p);
        self.coeffs = Some(coeffs);
        self.ndim = Some(ndim);
        self.order = Some(order);
        self.pieces = Some(pieces);
        self.is_valid = true;

        Ok(())
    }

    pub(crate) fn evaluate_spline(&self, xi: ArrayView1<'a, T>) -> Array<T, D> {
        unimplemented!();
    }
}
