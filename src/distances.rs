use num::Float;
use ndarray::ArrayView1;

pub fn squared_euclidean<T: Float + 'static>(a:ArrayView1<T>, b:ArrayView1<T>) -> T {
    let x = &a - &b; 
    x.dot(&x)
}