use ndarray::ArrayView1;
use num::Float;

pub fn squared_euclidean<T: Float + 'static>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {
    // let x = &a - &b;
    // x.dot(&x)

    a.into_iter()
        .zip(b.into_iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y))
}

pub fn squared_euclidean2<T: Float + 'static>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {
    let x = &a - &b;
    x.dot(&x)
}
