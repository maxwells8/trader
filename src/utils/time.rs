use chrono::{TimeZone, Utc, Local, Weekday, Datelike, Timelike};

pub fn market_open(timestamp: i64) -> bool {
    let dateTime = Utc.timestamp(timestamp, 0).with_timezone(&Local);
    match dateTime.weekday() {
        Weekday::Fri => dateTime.hour() < 15,
        Weekday::Sat => false,
        Weekday::Sun => dateTime.hour() > 15,
        _ => true
    }
}
