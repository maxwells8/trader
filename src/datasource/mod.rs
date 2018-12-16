use models::*;
use storage;
use chrono;
use std::cmp::max;
use std::sync::mpsc;
use redis::{self, Commands};
use json;

pub mod oanda;
pub mod external;
pub use self::oanda::OANDA;
use utils::*;

pub enum StreamType {
    Sim(Vec<Bar>),
    Live
}

pub trait DataSource {
    fn name() -> String;
    fn description(&self) -> String;
    fn granularity<'g, 's: 'g>(&'s self) -> &'g Granularity;
    fn fetch_latest_history(&self, count: usize) -> Vec<Bar>;
    fn fetch_history(&self, from: i64, to: i64) -> Vec<Bar>;
    fn poll_latest(&self) -> mpsc::Receiver<Bar>;

    fn load_from_storage(&self) -> Vec<Bar> {
        let file_name = format!("{}", self.description());
        match storage::load_bars(&Self::name(), &file_name) {
            Some(bars) => bars,
            None => {
                let bars = self.fetch_history(1104537600 /*January 1st 2005*/, chrono::Utc::now().timestamp()-5);
                storage::save_bars(&Self::name(), &file_name, &bars);
                bars
            }
        }
    }

    fn load_latest_history(&self, count: usize) -> Vec<Bar> {
        let bars = self.load_from_storage();
        bars[max(bars.len()-count, 0)..].to_vec()
    }

    fn load_history(&self, from: i64, to: i64) -> Vec<Bar> {
        println!("FROM: {}, TO: {}", from, to);
        let granularity = self.granularity().seconds() as i64;
        let mut start = from + (granularity - (from % granularity));
        if let Ok(client) = redis::Client::open("redis://127.0.0.1/") {
            if let Ok(con) = client.get_connection() {
                let mut bars = vec![];
                let mut fails = 0;
                while start < to {
                    let prefix = format!("{}_{}_", Self::name(), self.description());
                    //println!("Retrieving {}{}", &prefix, start);
                    if let Ok(json_bar) = con.get::<String, String>(format!("{}{}", prefix, start)) {
                        bars.push(json::from_str(&json_bar).expect("Failed to deserialize json"));
                        fails = 0;
                    } else {
                        if market_open(start) {
                            fails += 1;
                            let max_fails = 10;
                            if fails >= max_fails {
                                eprintln!("No bars were found for {} from Redis. Requesting data instead.", start);
                                for bar in self.fetch_history(start - (granularity * max_fails), to) {
                                    con.set::<String, String, ()>(format!("{}{}", prefix, bar.date), json::to_string(&bar).expect("Failed to serialize json")).unwrap();
                                    bars.push(bar);
                                }
                                break;
                            }
                        }
                    }
                    start += granularity;
                }
                return bars
            }
        }
        vec![]
    }

    fn load_history_old(&self, from: i64, to: i64) -> Vec<Bar> {
        let storage = self.load_from_storage();
        let mut bars = vec![];
        for bar in storage {
            if bar.date >= from && bar.date <= to {
                bars.push(bar);
            }
        };
        bars
    }
}
