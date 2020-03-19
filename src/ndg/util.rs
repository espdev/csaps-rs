use ndarray::Dimension;


pub(super) fn permute_axes<D>(ndim: usize) -> D
    where
        D: Dimension
{
    let mut permute_axes = D::zeros(ndim);

    permute_axes[0] = ndim - 1;
    for ax in 0..(ndim - 1) {
        permute_axes[ax + 1] = ax;
    }

    permute_axes
}
