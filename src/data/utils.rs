use data::models::Bar;
use super::super::Action;

pub fn calc_pips(open_price: f64, close_price: f64, action: &Action) -> f64 {
    (if action == &Action::Sell { open_price - close_price } else { close_price - open_price }) * 10000.0
}

fn find_l2_norm(bars: &[Bar]) -> Vec<f64> {
    let open = bars.iter().map(|bar| bar.open).map(|val| val.powi(2)).sum::<f64>().sqrt();
    let high = bars.iter().map(|bar| bar.high).map(|val| val.powi(2)).sum::<f64>().sqrt();
    let low = bars.iter().map(|bar| bar.low).map(|val| val.powi(2)).sum::<f64>().sqrt();
    let close = bars.iter().map(|bar| bar.close).map(|val| val.powi(2)).sum::<f64>().sqrt();
    let volume = bars.iter().map(|bar| bar.volume as f64).map(|val| val.powi(2)).sum::<f64>().sqrt();
    vec![open, high, low, close, volume]
}

pub fn normalize(bars: &[Bar]) -> Vec<Vec<f64>> {
    let norms = find_l2_norm(bars);
    bars.iter().map(|bar| {
        vec![bar.open / norms[0], bar.high / norms[1], bar.low / norms[2], bar.close / norms[3], bar.volume as f64 / norms[4]]
    }).collect()
}

pub trait Normalize {
    type Return;
    fn normalize(&self) -> Self::Return;
}

impl Normalize for Vec<Bar> {
    type Return = Vec<Vec<f64>>;

    fn normalize(&self) -> Self::Return {
        normalize(&self)
    }
}

impl Normalize for Vec<Vec<Bar>> {
    type Return = Vec<Vec<Vec<f64>>>;

    fn normalize(&self) -> Self::Return {
        self.iter().map(|vec| vec.normalize()).collect()
    }
}

impl Normalize for [Bar] {
    type Return = Vec<Vec<f64>>;

    fn normalize(&self) -> Self::Return {
        normalize(&self)
    }
}

pub trait Sample<T> {
    fn sample(&self, sample_size: usize, start: usize) -> Option<Vec<T>>;
    fn samples(&self, sample_size: usize) -> Vec<Vec<T>>;
    fn last_sample(&self, sample_size: usize) -> Option<Vec<T>>;
}

impl<T: Clone> Sample<T> for Vec<T> {

    fn sample(&self, sample_size: usize, start: usize) -> Option<Vec<T>> {
        if start + sample_size > self.len() { return None; }

        Some(self[start..start+sample_size].to_vec())
    }

    fn samples(&self, sample_size: usize) -> Vec<Vec<T>> {
        let mut samples = vec![];
        for index in 0..self.len() {
            if index >= sample_size {
                if let Some(sample) = self.sample(sample_size, index-sample_size) {
                    samples.push(sample);
                    continue;
                }
                break;
            }
        }
        samples
    }

    fn last_sample(&self, sample_size: usize) -> Option<Vec<T>> {
        if self.len() < sample_size { return None; }

        self.sample(sample_size, self.len() - sample_size)
    }
}