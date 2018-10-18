use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use hyper::header::Headers;
use reqwest::{self as http, Client, header, StatusCode, Response};
use json::{self, Value, Error};
use models::Granularity;
use models::Bar;
use std::str::FromStr;
use chrono;

header! { (AcceptDatetimeFormat, "Accept-Datetime-Format") => [String] }
header! { (ContentType, "Content-Type") => [String] }

pub struct OANDA {
    token: String,
    account: String
}

impl OANDA {
    const REST: &'static str = "https://api-fxpractice.oanda.com";
    const STREAM: &'static str = "https://stream-fxpractice.oanda.com";

    pub fn client() -> Self {
        let (token, account) = OANDA::load_credentials();
        OANDA {
            token,
            account
        }
    }

    fn load_credentials() -> (String, String) {
        if let Ok(mut cred_dir) = env::current_dir() {
            cred_dir.push("res/creds/oanda.json");
            if let Ok(file) = File::open(cred_dir.as_path()) {
                let creds: Result<Value, Error> = json::from_reader(BufReader::new(file));
                if let Ok(creds) = creds {
                    if let Some(token) = creds["token"].as_str() {
                        if let Some(account) = creds["account"].as_str() {
                            return (token.to_owned(), account.to_owned())
                        }
                    }
                }
            }
        }
        panic!("Failed to load OANDA credentials")
    }

    pub fn load_history_between(&self, instrument: &str, granularity: &Granularity, from: i64, to: i64) -> Vec<Bar> {
        let bar_count = ((to - from) as u32 / granularity.seconds()) as usize;
        self.load_history(instrument, granularity, bar_count, to)
    }

    pub fn load_latest_history(&self, instrument: &str, granularity: &Granularity, bar_count: usize) -> Vec<Bar> {
        self.load_history(instrument, granularity, bar_count, chrono::Utc::now().timestamp()-2)
    }

    fn load_history(&self, instrument: &str, granularity: &Granularity, bar_count: usize, mut epoch_seconds: i64) -> Vec<Bar> {
        let client = Client::new();
        let mut bars: Vec<Bar> = Vec::new();
        loop {
            let len = bars.len();
            if len >= bar_count { break; }

            let mut need = bar_count - len;
            println!("{} bars still needed", need);
            need += 1;
            let uri = format!("{}/v3/instruments/{}/candles?to={}&granularity={:?}&price=B&count={}&smooth=true&includeLast=false",
                              OANDA::REST,
                              &instrument,
                              &epoch_seconds,
                              &granularity,
                              if need > 5000 { 5000 } else { need });
            println!("{}", &uri);
            match client.get(&uri)
                .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                .header(AcceptDatetimeFormat("UNIX".to_owned()))
                .header(ContentType("application/json".to_owned()))
                .send() {
                Ok(mut res) => {
                    if let Some(mut res_bars) = OANDA::parse_history(&mut res) {
                        res_bars.reverse();
                        for bar in res_bars {
                            epoch_seconds = bar.date;
                            if bar.complete {
                                bars.push(bar);
                            }
                        }
                    }
                    else {
                        eprintln!("OANDA ERROR PARSING HISTORY: {:?}", &res);
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("OANDA ERROR LOADING HISTORY: {}", e);
                    break;
                }
            }
        }
        bars.reverse();
        while bars.len() > bar_count {
            bars.remove(0);
        }
        bars
    }

    fn parse_history(res: &mut Response) -> Option<Vec<Bar>> {
        if res.status().is_success() {
            if let Ok(value) = res.json::<Value>() {
                //TODO Verify symbol and granularity?
                if let Some(candles) = value["candles"].as_array() {
                    if candles.len() > 0 {
                        return Some(candles.iter().map(|candle| {
                            let bid = &candle["bid"];
                            Bar {
                                open: bid["o"].parse_as_f64().unwrap(),
                                high: bid["h"].parse_as_f64().unwrap(),
                                low: bid["l"].parse_as_f64().unwrap(),
                                close: bid["c"].parse_as_f64().unwrap(),
                                volume: candle["volume"].as_i64().unwrap() as i32,
                                complete: candle["complete"].as_bool().unwrap(),
                                date: candle["time"].parse_as_f64().unwrap() as i64
                            }
                        }).collect());
                    }
                }
            }
        }
        None
    }

    fn open_trades_url(&self) -> String {
        format!("{}/v3/accounts/{}/openTrades", OANDA::REST, &self.account)
    }

    fn order_url(&self) -> String {
        format!("{}/v3/accounts/{}/orders", OANDA::REST, &self.account)
    }

    fn stream_url(&self, instrument: &str) -> String {
        format!("{}/v3/accounts/{}/pricing/stream?instruments={}", OANDA::STREAM, &self.account, instrument)
    }
}

trait ParseValue {
    fn parse_as<T: FromStr>(&self) -> Option<T>;
    fn parse_as_f64(&self) -> Option<f64>;
    fn parse_as_i32(&self) -> Option<i32>;
}

impl ParseValue for Value {
    fn parse_as<T: FromStr>(&self) -> Option<T> {
        if let Some(s) = self.as_str() {
            if let Ok(f) = s.parse() {
                return Some(f);
            }
        }
        None
    }

    fn parse_as_f64(&self) -> Option<f64> {
        self.parse_as::<f64>()
    }

    fn parse_as_i32(&self) -> Option<i32> {
        self.parse_as::<i32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_creds() {
        OANDA::load_credentials();
    }

    #[test]
    fn load_history() {
        let oanda = OANDA::client();
        let count = 7000;
        let bars = oanda.load_latest_history("EUR_USD", &Granularity::M15, count);
        assert_eq!(count, bars.len());
        let mut last = 0;
        for current in 1..bars.len() {
            //print!("{} ", current);
            let diff = ((bars[current].date - bars[last].date) / 60) / 15;
            //println!("{}", diff);
            assert!(diff == 1 || diff == 2 || diff == 193);
            last = current;
        }
    }

    #[test]
    fn load_history_between() {
        let oanda = OANDA::client();
        let from = 1532388304;
        let to = 1538688304;
        let bars = oanda.load_history_between("EUR_USD", &Granularity::M15, from, to);
        assert_eq!(7000, bars.len());
        let mut last = 0;
        for current in 1..bars.len() {
            //print!("{} ", current);
            let diff = ((bars[current].date - bars[last].date) / 60) / 15;
            //println!("{}", diff);
            assert!(diff == 1 || diff == 2 || diff == 193);
            last = current;
        }
    }
}
