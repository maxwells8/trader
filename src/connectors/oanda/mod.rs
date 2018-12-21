use models::{Granularity, Bar, PositionType, Position, AccountInfo, Trade};

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;
use std::sync::mpsc;
use std::time;
use std::thread;

use reqwest::{Client, header, Response};
use json::{self, Value, Error};
use chrono;

mod utils; use self::utils::*;

#[cfg(test)]
mod tests;

header! { (AcceptDatetimeFormat, "Accept-Datetime-Format") => [String] }
header! { (ContentType, "Content-Type") => [String] }

pub struct OANDA {
    token: String,
    account: String
}

impl OANDA {
    pub fn client() -> Self {
        let (token, account) = load_credentials();
        OANDA {
            token,
            account
        }
    }

    pub fn load_history_between(&self, instrument: &str, granularity: &Granularity, from: i64, to: i64) -> Vec<Bar> {
        Self::load_history_between_internal(&self.token, instrument, granularity, from, to)
    }

    pub fn load_latest_history(&self, instrument: &str, granularity: &Granularity, bar_count: usize) -> Vec<Bar> {
        Self::load_latest_history_internal(&self.token, instrument, granularity, bar_count)
    }

    fn load_history_between_internal(token: &str, instrument: &str, granularity: &Granularity, mut from: i64, to: i64) -> Vec<Bar> {
        let mut bars = vec![];
        let seconds = granularity.seconds() as i64;
        let timestep = seconds * 5000;
        let mut req_to;
        loop {
            if from > to - seconds { break; }

            req_to = from + timestep;
            if req_to > to { req_to = to }

            let res_bars = Self::load_history(token, history_between_url(instrument, granularity, from, req_to));
            if res_bars.is_empty() { break; }
            for bar in res_bars {
                let next = bar.date + seconds;
                if next > from {
                    from = next;
                    if bar.complete {
                        bars.push(bar);
                    }
                }
                else { return bars; }
            }
        }
        bars
    }

    fn load_latest_history_internal(token: &str, instrument: &str, granularity: &Granularity, bar_count: usize) -> Vec<Bar> {
        let mut epoch_seconds = chrono::Utc::now().timestamp()-5;
        let mut bars: Vec<Bar> = Vec::new();
        loop {
            let len = bars.len();
            if len >= bar_count { break; }

            let mut need = bar_count - len;
            debug!("{} bars still needed", need);
            need += 1;
            let req_count = if need > 5000 { 5000 } else { need };
            let mut res_bars = Self::load_history(token, history_count_to_url(instrument, granularity, req_count, epoch_seconds));
            res_bars.reverse();
            for bar in res_bars {
                epoch_seconds = bar.date;
                if bar.complete {
                    bars.push(bar);
                }
            }
        }
        bars.reverse();
        while bars.len() > bar_count {
            bars.remove(0);
        }
        bars
    }

    fn load_history(token: &str, uri: String) -> Vec<Bar> {
        debug!("{}", &uri);
        match Client::new().get(&uri)
            .header(header::Authorization(header::Bearer { token: String::from(token) }))
            .header(AcceptDatetimeFormat("UNIX".to_owned()))
            .header(ContentType("application/json".to_owned()))
            .send() {
            Ok(mut res) => {
                if let Some(mut res_bars) = parse_history(&mut res) {
                    return res_bars;
                }
                else {
                    if !res.status().is_success() {
                        error!("OANDA ERROR PARSING HISTORY: {:?}", &res);
                    }
                }
            }
            Err(e) => {
                error!("OANDA ERROR LOADING HISTORY: {}", e);
            }
        };
        vec![]
    }

    pub fn poll_latest(&self, instrument: &str, granularity: &Granularity) -> mpsc::Receiver<Bar> {
        let (channel, recv) = mpsc::channel();
        let instrument = String::from(instrument);
        let granularity = granularity.clone();
        let token = self.token.clone();
        thread::spawn(move || {
            let mut history = vec![];
            loop {
                //Get the most recent two bars
                let bars = OANDA::load_latest_history_internal(&token, &instrument, &granularity, 2);
                if bars.is_empty() { continue; }

                if history.is_empty() {
                    for bar in &bars {
                        if bar.complete {
                            history.push(bar.clone());
                        }
                    }
                }
                else {
                    let last_timestamp = history.last().unwrap().date;

                    //If not add them and send them through the channel
                    for bar in &bars {
                        if bar.date > last_timestamp && bar.complete {
                            history.push(bar.clone());
                            &channel.send(bar.clone());
                        }
                    }
                }

                //Check time of next incomplete bar
                if let &Some(current) = &bars.last() {
                    let now = chrono::Utc::now().timestamp();
                    let mut sleep_time = current.date + ((granularity.seconds() * 2) as i64) - now;
                    if sleep_time < 0 {
                        sleep_time = 2
                    }
                    //println!("Now: {}, Bar: {}, Sleep: {}", now, current.date, sleep_time);
                    if sleep_time > 0 {
                        thread::sleep(time::Duration::from_secs(sleep_time as u64));
                    }
                }
            }
        });
        recv
    }

    pub fn submit_order(&self, instrument: &str, pos: &PositionType, units: u32) -> bool {
        let body = json!({
            "order": {
                "type": "MARKET",
                "timeInForce": "FOK",
                "instrument": &instrument,
                "units": if pos == &PositionType::Long { units as i32 } else { (units as i32) * -1 }
            }
        }).to_string();
        let client = Client::new();
        match client.post(&order_url(&self.account))
            .header(header::Authorization(header::Bearer { token: self.token.clone() }))
            .header(AcceptDatetimeFormat("UNIX".to_owned()))
            .header(ContentType("application/json".to_owned()))
            .body(body)
            .send() {
            Ok(mut res) => {
                if res.status().is_success() {
                    return true
                    // let body = res.json::<Value>();
                    // if let Ok(body) = body {
                    //     let trans = &body["orderFillTransaction"];
                    //     return Some(Trade {
                    //         id: trans["id"].parse_as().unwrap(),
                    //         date: trans["time"].parse_as_f64().unwrap() as i64,
                    //         instrument: String::from(instrument),
                    //         position: pos.clone(),
                    //         open_price: trans["price"].parse_as_f64().unwrap(),
                    //         units,
                    //         closed: vec![]
                    //     });
                    // }
                }
                else {
                    error!("OANDA ORDER STATUS ERROR: {:?}", &res);
                }
            },
            Err(e) => {
                error!("OANDA ORDER ERROR: {:?}", e);
            }
        };
        false
    }

    pub fn fetch_position(&self, instrument: &str) -> Result<Option<Position>, ()> {
        let client = Client::new();
        match client.get(&open_position_url(&self.account, instrument))
                    .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                    .header(AcceptDatetimeFormat("UNIX".to_owned()))
                    .header(ContentType("application/json".to_owned()))
                    .send() {
            Ok(mut res) => {
                let body = res.json::<Value>();
                if let Ok(body) = body {
                    let position = &body["position"];
                    let long = &position["long"];
                    let mut units = long["units"].parse_as::<u32>().unwrap_or(0);
                    if units > 0 {
                        return Ok(Some(Position {
                            pos_type: PositionType::Long,
                            instrument: String::from(instrument),
                            avg_price: long["averagePrice"].parse_as_f64().unwrap(),
                            units
                        }))
                    }
                    else {
                        let short = &position["short"];
                        units = short["units"].parse_as::<u32>().unwrap_or(0);
                        if units > 0 {
                            return Ok(Some(Position {
                                pos_type: PositionType::Short,
                                instrument: String::from(instrument),
                                avg_price: short["averagePrice"].parse_as_f64().unwrap(),
                                units
                            }))
                        } else {
                            return Ok(None)
                        }
                    }
                }
            },
            Err(e) => {
                error!("OANDA POSITION ERROR: {:?}", e);
            }
        };
        Err(())
    }

    pub fn fetch_account_info(&self) -> Result<AccountInfo, ()> {
        let client = Client::new();
        match client.get(&account_summary_url(&self.account))
                    .header(header::Authorization(header::Bearer { token: self.token.clone() }))
                    .header(AcceptDatetimeFormat("UNIX".to_owned()))
                    .header(ContentType("application/json".to_owned()))
                    .send() {
            Ok(mut res) => {
                let body = res.json::<Value>();
                if let Ok(body) = body {
                    let account = &body["account"];
                    return Ok(AccountInfo {
                        balance: account["balance"].parse_as_f64().unwrap(),
                        unrealized_balance: account["NAV"].parse_as_f64().unwrap(),
                        margin_rate: account["marginRate"].parse_as_f64().unwrap(),
                        margin_available: account["marginAvailable"].parse_as_f64().unwrap(),
                        margin_used: account["marginUsed"].parse_as_f64().unwrap(),
                        unrealized_pl: account["unrealizedPL"].parse_as_f64().unwrap()
                    })
                }
            },
            Err(e) => {
                error!("OANDA ACCOUNT SUMMMARY ERROR: {:?}", e);
            }
        };
        Err(())
    }
}
