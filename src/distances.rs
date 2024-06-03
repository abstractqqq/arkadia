use ndarray::ArrayView1;
use num::Float;

pub fn squared_euclidean<T: Float + 'static>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {

    a.into_iter()
        .zip(b.into_iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + (x - y) * (x - y))

}


pub fn linf_dist<T: Float + 'static>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {

    a.into_iter()
        .zip(b.into_iter())
        .fold(T::min_value(), |acc, (&x, &y)| acc.max((x - y).abs()))

}

pub fn l1_dist<T: Float + 'static>(a: ArrayView1<T>, b: ArrayView1<T>) -> T {

    a.into_iter()
        .zip(b.into_iter())
        .fold(T::min_value(), |acc, (&x, &y)| acc + (x - y).abs())

}