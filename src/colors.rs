
macro_rules! win_loss {
    ($e:expr) => {{
        let val = format!("{:.2}", $e);
        if $e > 0.0 {
            ansi_term::Color::Green.paint(val)
        }
        else if $e < 0.0 {
            ansi_term::Color::Red.paint(val)
        }
        else {
            ansi_term::Color::White.paint(val)
        }
    }};
}

macro_rules! avg {
    ($e:expr) => {{
        let val = format!("{:.2}", $e);
        if $e > 50.0 {
            ansi_term::Color::Green.paint(val)
        }
        else if $e < 50.0 {
            ansi_term::Color::Red.paint(val)
        }
        else {
            ansi_term::Color::White.paint(val)
        }
    }};
}

macro_rules! red {
    ($e:expr) => { ansi_term::Color::Red.paint(format!("{}", $e)) };
}

macro_rules! green {
    ($e:expr) => { ansi_term::Color::Green.paint(format!("{}", $e)) };
}

macro_rules! blue {
    ($e:expr) => { ansi_term::Color::Blue.paint(format!("{}", $e)) };
}

macro_rules! purple {
    ($e:expr) => { ansi_term::Color::Purple.paint(format!("{}", $e)) };
}

macro_rules! cyan {
    ($e:expr) => { ansi_term::Color::Cyan.paint(format!("{}", $e)) };
}

macro_rules! white {
    ($e:expr) => { ansi_term::Color::White.paint(format!("{}", $e)) };
}