
use super::models::*;

pub trait Normalizer {
    fn normalizer(&self) -> f64;
}

impl Normalizer for [Bar] {
    fn normalizer(&self) -> f64 {
        let mut min: f64 = self[0].low;
        let mut max: f64 = self[0].high;
        for bar in self {
            let low = bar.low;
            let high = bar.high;
            min = if min > low { low } else { min };
            max = if max < high { high } else { max };
        }
        (min.powi(2) + max.powi(2)).sqrt()
    }
}

//TODO GET RID OF THESE
pub trait Normalize {
    fn normalize(&self, batch_size: usize) -> Vec<Vec<[f64; 4]>>;
}

impl Normalize for Series {
    fn normalize(&self, batch_size: usize) -> Vec<Vec<[f64; 4]>> {
        //TODO Instead, this should be dividing by the standard deviation
        let mut normalized: Vec<Vec<[f64; 4]>> = Vec::new();
        let bars = &self.bars;
        for x in batch_size..&bars.len()+1 {
            let slice: &[Bar] = &bars[x - batch_size..x];
            let norm = slice.normalizer();
            let slice = slice.iter().map(|bar| {
                [bar.open/norm, bar.high/norm, bar.low/norm, bar.close/norm]
            }).collect();
            normalized.push(slice);
        }
        normalized
    }
}