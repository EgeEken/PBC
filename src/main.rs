mod palette;
mod pbc3;
mod types;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

use crate::pbc3::PBC3;

#[derive(Parser, Debug)]
#[command(name = "pbc", version)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[command(name = "decode", alias = "dec", alias = "d")]
    Decode {
        #[arg(required = true)]
        file_input: String,

        #[arg(short = 't', long = "timed", default_value_t = false)]
        timed: bool,
    },
}

fn output_path_for(input: &str) -> PathBuf {
    let path = PathBuf::from(input);
    if path.extension().is_some() {
        path.with_extension("png")
    } else {
        PathBuf::from(format!("{input}.png"))
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Commands::Decode { file_input, timed } => {
            let result = PBC3::decompress(std::fs::read(&file_input)?, None)?;
            let output = output_path_for(&file_input);
            PBC3::save_png(&result, &output)?;
            if timed {
                println!(
                    "decoded in {:.3}s -> {}",
                    result.encode_seconds,
                    output.display()
                );
            } else {
                println!("wrote {}", output.display());
            }
        }
    }
    Ok(())
}
