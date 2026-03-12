use comfy_table::{Cell, Table};

use gguf_inspect::types::*;

/// Format bytes into a human-readable string.
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format a parameter count (e.g. 7_000_000_000 -> "7.00B").
fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1e3)
    } else {
        format!("{count}")
    }
}

/// Print the summary overview of a GGUF file.
pub fn print_summary(gguf: &GgufFile) {
    let arch = gguf.architecture().unwrap_or("unknown");

    println!("GGUF Model Summary");
    println!("{}", "=".repeat(50));

    if let Some(name) = gguf.model_name() {
        println!("  Model name:       {name}");
    }
    println!("  Architecture:     {arch}");
    println!("  GGUF version:     {}", gguf.version);
    println!("  Tensor count:     {}", gguf.tensors.len());
    println!(
        "  Parameters:       {}",
        format_params(gguf.total_parameters())
    );

    if let Some(val) = gguf.arch_metadata("context_length") {
        println!("  Context length:   {val}");
    }
    if let Some(val) = gguf.arch_metadata("embedding_length") {
        println!("  Embedding size:   {val}");
    }
    if let Some(val) = gguf.arch_metadata("block_count") {
        println!("  Layers:           {val}");
    }
    if let Some(val) = gguf.arch_metadata("attention.head_count") {
        println!("  Attention heads:  {val}");
    }
    if let Some(val) = gguf.arch_metadata("attention.head_count_kv") {
        println!("  KV heads:         {val}");
    }

    // Vocab size from tokenizer tokens array
    if let Some(tokens) = gguf.get_metadata("tokenizer.ggml.tokens") {
        if let Some(arr) = tokens.as_array() {
            println!("  Vocab size:       {}", arr.len());
        }
    }

    if let Some(ft) = gguf.get_metadata("general.file_type") {
        if let Some(ft_val) = ft.as_u32() {
            println!("  Quantization:     {}", file_type_name(ft_val));
        }
    }

    println!("  File size:        {}", format_bytes(gguf.file_size));
    println!(
        "  Tensor data size: {}",
        format_bytes(gguf.total_tensor_size())
    );

    // Estimated memory: tensor data + ~10% overhead for runtime buffers
    let estimated_mem = (gguf.total_tensor_size() as f64 * 1.1) as u64;
    println!("  Est. memory:      {} (tensors + ~10% overhead)", format_bytes(estimated_mem));
}

/// Print all raw metadata key-value pairs.
pub fn print_metadata(gguf: &GgufFile) {
    println!("\nMetadata ({} entries)", gguf.metadata.len());
    println!("{}", "=".repeat(50));

    let mut table = Table::new();
    table.set_header(vec!["Key", "Value"]);

    for (key, value) in &gguf.metadata {
        let display = match value {
            GgufValue::Array(arr) => {
                if arr.len() > 5 {
                    let first: Vec<String> = arr[..3].iter().map(|v| format!("{v}")).collect();
                    format!("[{}, ... ({} total)]", first.join(", "), arr.len())
                } else {
                    format!("{value}")
                }
            }
            other => format!("{other}"),
        };
        table.add_row(vec![Cell::new(key), Cell::new(display)]);
    }

    println!("{table}");
}

/// Print tensor info table.
pub fn print_tensors(gguf: &GgufFile) {
    println!("\nTensors ({} entries)", gguf.tensors.len());
    println!("{}", "=".repeat(50));

    let mut table = Table::new();
    table.set_header(vec!["Name", "Shape", "Type", "Size"]);

    for tensor in &gguf.tensors {
        table.add_row(vec![
            Cell::new(&tensor.name),
            Cell::new(tensor.shape_str()),
            Cell::new(format!("{}", tensor.dtype)),
            Cell::new(format_bytes(tensor.size_bytes())),
        ]);
    }

    println!("{table}");
}

/// Print JSON output of the entire parsed file.
pub fn print_json(gguf: &GgufFile) {
    // Build a JSON-friendly structure
    let output = serde_json::json!({
        "version": gguf.version,
        "tensor_count": gguf.tensors.len(),
        "metadata_count": gguf.metadata.len(),
        "architecture": gguf.architecture(),
        "model_name": gguf.model_name(),
        "total_parameters": gguf.total_parameters(),
        "file_size": gguf.file_size,
        "tensor_data_size": gguf.total_tensor_size(),
        "metadata": gguf.metadata.iter()
            .map(|(k, v)| serde_json::json!({"key": k, "value": v}))
            .collect::<Vec<_>>(),
        "tensors": gguf.tensors.iter()
            .map(|t| serde_json::json!({
                "name": t.name,
                "shape": t.dimensions,
                "type": format!("{}", t.dtype),
                "elements": t.element_count(),
                "size_bytes": t.size_bytes(),
            }))
            .collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
