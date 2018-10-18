pub mod models;
pub mod process;
pub mod utils;

use self::models::{Granularity, Series};
use json;
use std::path::{Path, PathBuf};
use std::io::{Write, BufReader};
use std::fs::File;

#[macro_export]
macro_rules! run {
    ($expr:expr, $str:expr) => (match $expr {
        Ok(val) => val,
        Err(error) => return Err(format!("{}: {:?}", $str, error)),
    })
}

pub trait Api {
    fn load(&mut self, symbol: &str, granularity: &Granularity, count: i32) -> Result<Series, String>;
    fn load_since(&mut self, symbol: &str, granularity: &Granularity, mut epoch_seconds: i64) -> Result<Series, String>;
}

pub struct Loader<T: Api> {
    pub api: T
}

impl<T: Api> Loader<T> {
    pub fn test(&mut self) -> Result<(), String> {
        let series: Series = run!(self.api.load_since("GBP_USD", &Granularity::M5, 1), "Failed to load Series from Api");
        println!("{}", series.bars.len());
        self.save(series)
    }

    pub fn download(&mut self, symbol: &str, granularity: Granularity) -> Result<(), String> {
        let series = run!(self.api.load(symbol, &granularity, 5000), "Failed to load Series from Api");
        self.save(series)
    }

    pub fn save(&self, series: Series) -> Result<(), String> {
        let json = run!(json::to_string(&series), "Failed to serialize json");
        let path = data_path(&series.symbol, &series.granularity);
        let mut file = run!(File::create(&path), "Failed to open file");
        run!(file.write_all(json.as_bytes()), "Failed to write to file");
        Ok(())
    }

    pub fn load(&self, symbol: &str, granularity: Granularity) -> Result<Series, String> {
        let path = data_path(symbol, &granularity);
        let file = run!(File::open(&path), "Failed to open file");
        let data: Series = run!(json::from_reader(BufReader::new(file)), "Failed to deserialize json");
        Ok(data)
    }
}

impl Series {
    pub fn get_labels(&self) -> Result<Vec<Vec<i32>>, String> {
        let path = label_path(&self.symbol, &self.granularity);
        let file = run!(File::open(&path), "Failed to open file");
        let data: Vec<String> = run!(json::from_reader(BufReader::new(file)), "Failed to deserialize json");
        //One hot encode hack
        let data = data.iter().map(|label| {
            match label.as_ref() {
                "Buy" => vec!(1, 0, 0),
                "Sell" => vec!(0, 1, 0),
                _ => vec!(0, 0, 1)
            }
        }).collect();
        Ok(data)
    }
}

fn data_path(symbol: &str, granularity: &Granularity) -> PathBuf {
    let file_name = format!("res/series/{}_{:?}.json", symbol, granularity);
    Path::new(&file_name).to_owned()
}

fn label_path(symbol: &str, granularity: &Granularity) -> PathBuf {
    let file_name = format!("res/labels/{}_{:?}.json", symbol, granularity);
    Path::new(&file_name).to_owned()
}