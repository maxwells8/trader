use std::fs::{File, self};
use std::io::{Read, stdout, Write};
use std::path::Path;
use std::env;
use std::process::{Command, ExitStatus};

use connectors::Connector;

pub mod bar;
pub mod external;

pub trait Trainer<'c, C: 'c + Connector> {
    fn train(&mut self, connector: &'c C);
}