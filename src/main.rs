mod display;
mod parser;
mod types;

use clap::Parser;
use std::path::PathBuf;
use std::process;

#[derive(Parser)]
#[command(
    name = "gguf-inspect",
    about = "Inspect GGUF model files and print metadata, tensor info, and architecture details"
)]
struct Cli {
    /// Path to the GGUF file
    file: PathBuf,

    /// List all tensors with shapes, types, and sizes
    #[arg(long)]
    tensors: bool,

    /// Dump all raw metadata key-value pairs
    #[arg(long)]
    metadata: bool,

    /// Output in JSON format
    #[arg(long)]
    json: bool,
}

fn main() {
    let cli = Cli::parse();

    if !cli.file.exists() {
        eprintln!("Error: file not found: {}", cli.file.display());
        process::exit(1);
    }

    let gguf = match parser::parse_gguf(&cli.file) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Error parsing GGUF file: {e}");
            process::exit(1);
        }
    };

    if cli.json {
        display::print_json(&gguf);
        return;
    }

    display::print_summary(&gguf);

    if cli.metadata {
        display::print_metadata(&gguf);
    }

    if cli.tensors {
        display::print_tensors(&gguf);
    }
}
