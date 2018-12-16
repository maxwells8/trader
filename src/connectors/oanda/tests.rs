use super::*;

#[test]
fn load_creds() {
    load_credentials();
}

#[test]
fn load_history() {
    let oanda = OANDA::client();
    let count = 7000;
    let bars = oanda.load_latest_history("EUR_USD", &Granularity::M15, count);
    assert_eq!(count, bars.len());
    let mut last = 0;
    for current in 1..bars.len() {
        //print!("{:?}", bars[current]);
        let diff = ((bars[current].date - bars[last].date) / 60) / 15;
        //println!("{}", diff);
        assert!(diff == 1 || diff == 2 || diff == 193 || diff == 197);
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
        //print!("{} ", bars[current].spread);
        let diff = ((bars[current].date - bars[last].date) / 60) / 15;
        //println!("{}", diff);
        assert!(diff == 1 || diff == 2 || diff == 193);
        last = current;
    }
}

#[test]
fn poll_latest() {
    let oanda = OANDA::client();
    let recv = oanda.poll_latest("EUR_USD", &Granularity::M1);

    for bar in recv.iter() {
        println!("{:?}", &bar);
    }
}

#[test]
fn instrument_details() {
    let (token, account) = load_credentials();
    let client = Client::new();
    match client.get(&instrument_details_url(&account, "EUR_USD"))
                .header(header::Authorization(header::Bearer { token }))
                .header(ContentType("application/json".to_owned()))
                .send() {
        Ok(mut res) => {
            println!("{:?}", res.json::<Value>())
        },
        Err(_) => {}
    };
}

#[test]
fn place_trade() {
    let client = OANDA::client();
    let trade = client.submit_order("EUR_USD", &PositionType::Long, 100);
    println!("{:?}", trade);
    let trade = client.submit_order("EUR_USD", &PositionType::Short, 100);
    println!("{:?}", trade);
}

#[test]
fn fetch_position() {
    let client = OANDA::client();
    println!("{:?}", client.fetch_position("EUR_USD"));
}

#[test]
fn fetch_account_info() {
    let client = OANDA::client();
    println!("{:?}", client.fetch_account_info());
}
