use json;
use models::Granularity;
use nd;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::io::Write;
use models::Bar;

pub fn file_exists(folder: &str, label: &str) -> bool {
    let file_name = &format!("res/series/{}/{}.json", folder, label);
    let path = Path::new(file_name);
    path.exists()
}

pub fn load_bars(folder: &str, label: &str) -> Option<Vec<Bar>> {
    let file_name = &format!("res/series/{}/{}.json", folder, label);
    if let Ok(file) = File::open(Path::new(&file_name)) {
        if let Ok(bars) = json::from_reader(BufReader::new(file)) {
            return Some(bars)
        }
    }
    None
    /*
    if let Ok(file) = File::open(Path::new(&file_name)) {
        if let Ok(obj) = json::from_reader::<_, json::Value>(BufReader::new(file)) {
            let bars = obj.as_array().unwrap()
               .iter()
               .map(|bar| [
                   bar["open"].as_f64().unwrap(),
                   bar["high"].as_f64().unwrap(),
                   bar["low"].as_f64().unwrap(),
                   bar["close"].as_f64().unwrap(),
                   bar["volume"].as_i64().unwrap() as f64,
                   bar["date"].as_i64().unwrap() as f64
               ]).collect::<Vec<[f64; 6]>>();
            println!("{}", bars.len());
            return Some(nd::arr2(bars.as_ref()));
        }
    }
    None
    */
}

pub fn save_bars(folder: &str, label: &str, bars: &Vec<Bar>) {
    let file_name = &format!("res/series/{}/{}.json", folder, label);
    let path = Path::new(file_name);
    let json = json::to_string(&bars).expect("Failed to serialize json");
    let mut file = File::create(path).expect("Failed to open file");
    file.write_all(json.as_bytes()).expect("Failed to write to file");
}