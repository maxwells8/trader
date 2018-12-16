use super::*;

pub fn load_credentials() -> (String, String) {
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

pub trait ParseValue {
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

pub fn parse_history(res: &mut Response) -> Option<Vec<Bar>> {
    if res.status().is_success() {
        if let Ok(value) = res.json::<Value>() {
            if let Some(candles) = value["candles"].as_array() {
                if candles.len() > 0 {
                    return Some(candles.iter().map(move |candle| {
                        let date = candle["time"].parse_as_f64().unwrap() as i64;
                        let bid = &candle["bid"];
                        let ask = &candle["ask"];
                        let bid_close = bid["c"].parse_as_f64().unwrap();
                        let ask_close = ask["c"].parse_as_f64().unwrap();
                        Bar {
                            open: bid["o"].parse_as_f64().unwrap(),
                            high: bid["h"].parse_as_f64().unwrap(),
                            low: bid["l"].parse_as_f64().unwrap(),
                            close: bid_close,
                            volume: candle["volume"].as_i64().unwrap() as i32,
                            spread: ask_close - bid_close,
                            complete: candle["complete"].as_bool().unwrap(),
                            date
                        }
                    }).collect());
                }
            }
        }
    }
    None
}

const REST: &'static str = "https://api-fxpractice.oanda.com";
const STREAM: &'static str = "https://stream-fxpractice.oanda.com";

pub fn history_count_to_url(instrument: &str, granularity: &Granularity, bar_count: usize, epoch_seconds: i64) -> String {
    return format!("{}&to={}&count={}",
                    history_base_url(instrument, granularity),
                    &epoch_seconds,
                    bar_count);
}

pub fn history_between_url(instrument: &str, granularity: &Granularity, from: i64, to: i64) -> String {
    return format!("{}&from={}&to={}",
                    history_base_url(instrument, granularity),
                    from,
                    to);
}

fn history_base_url(instrument: &str, granularity: &Granularity) -> String {
    return format!("{}/v3/instruments/{}/candles?&granularity={:?}&price=BA&smooth=true&includeLast=false",
                    REST,
                    &instrument,
                    &granularity);
}

pub fn instrument_details_url(account: &str, instrument: &str) -> String {
    format!("{}/v3/accounts/{}/instruments?instruments={}", REST, account, instrument)
}

pub fn open_trades_url(account: &str) -> String {
    format!("{}/v3/accounts/{}/openTrades", REST, account)
}

pub fn order_url(account: &str) -> String {
    format!("{}/v3/accounts/{}/orders", REST, account)
}

pub fn open_position_url(account: &str, instrument: &str) -> String {
    format!("{}/v3/accounts/{}/positions/{}", REST, account, instrument)
}

pub fn account_summary_url(account: &str) -> String {
    format!("{}/v3/accounts/{}/summary", REST, account)
}

pub fn stream_url(account: &str, instrument: &str) -> String {
    format!("{}/v3/accounts/{}/pricing/stream?instruments={}", STREAM, account, instrument)
}
