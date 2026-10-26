#[path = "../palette.rs"]
mod palette;
#[path = "../pbc3.rs"]
mod pbc3;
#[path = "../types.rs"]
mod types;
use anyhow::Result;
use std::fs;
use std::io::{self, BufRead, Write};
use std::time::Instant;
use pbc3::PBC3;

fn main() -> Result<()> {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    let streams: Vec<Vec<u8>> = paths.iter().map(fs::read).collect::<std::io::Result<_>>()?;
    for data in &streams { PBC3::decompress(data, None)?; }
    println!("READY");
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let index: usize = line?.trim().parse()?;
        let start = Instant::now();
        let result = PBC3::decompress(&streams[index], None)?;
        println!("{:.9}\t{:.9}\t{}\t{}", result.encode_seconds, start.elapsed().as_secs_f64(), result.image_width, result.image_height);
        io::stdout().flush()?;
    }
    Ok(())
}
