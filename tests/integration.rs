//! Integration tests: write a minimal GGUF file, parse it, and verify.

use byteorder::{LittleEndian, WriteBytesExt};
use std::io::Write;
use std::path::Path;

/// Write a minimal valid GGUF v3 file with metadata and tensors.
fn create_test_gguf(path: &Path) {
    let mut buf: Vec<u8> = Vec::new();

    // --- Header ---
    // Magic: "GGUF" in little-endian = 0x46554747
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    // Version 3
    buf.write_u32::<LittleEndian>(3).unwrap();
    // Tensor count: 2
    buf.write_u64::<LittleEndian>(2).unwrap();
    // Metadata KV count: 3
    buf.write_u64::<LittleEndian>(3).unwrap();

    // --- Metadata ---
    // KV 1: general.architecture = "test" (string, type 8)
    write_string(&mut buf, "general.architecture");
    buf.write_u32::<LittleEndian>(8).unwrap(); // String type
    write_string(&mut buf, "test");

    // KV 2: general.name = "test-model" (string, type 8)
    write_string(&mut buf, "general.name");
    buf.write_u32::<LittleEndian>(8).unwrap();
    write_string(&mut buf, "test-model");

    // KV 3: test.block_count = 2 (uint32, type 4)
    write_string(&mut buf, "test.block_count");
    buf.write_u32::<LittleEndian>(4).unwrap(); // Uint32 type
    buf.write_u32::<LittleEndian>(2).unwrap();

    // --- Tensor info ---
    // Tensor 1: "weight_a", 2D [4, 8], F32 (type 0), offset 0
    write_string(&mut buf, "weight_a");
    buf.write_u32::<LittleEndian>(2).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(4).unwrap(); // dim 0
    buf.write_u64::<LittleEndian>(8).unwrap(); // dim 1
    buf.write_u32::<LittleEndian>(0).unwrap(); // dtype = F32
    buf.write_u64::<LittleEndian>(0).unwrap(); // offset

    // Tensor 2: "bias_b", 1D [8], F16 (type 1), offset 128
    write_string(&mut buf, "bias_b");
    buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims
    buf.write_u64::<LittleEndian>(8).unwrap(); // dim 0
    buf.write_u32::<LittleEndian>(1).unwrap(); // dtype = F16
    buf.write_u64::<LittleEndian>(128).unwrap(); // offset

    // Pad to alignment and write dummy tensor data.
    // The parser does not read actual tensor data, so just pad the file.
    let padding = 256;
    buf.resize(buf.len() + padding, 0u8);

    std::fs::write(path, &buf).unwrap();
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    buf.write_all(s.as_bytes()).unwrap();
}

/// Write a file with invalid magic bytes.
fn create_invalid_magic_gguf(path: &Path) {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(0xDEADBEEF).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap();
    std::fs::write(path, &buf).unwrap();
}

/// Write a file with unsupported version.
fn create_bad_version_gguf(path: &Path) {
    let mut buf: Vec<u8> = Vec::new();
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(99).unwrap(); // unsupported version
    buf.write_u64::<LittleEndian>(0).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap();
    std::fs::write(path, &buf).unwrap();
}

/// Write a GGUF with various metadata value types for coverage.
fn create_multi_type_gguf(path: &Path) {
    let mut buf: Vec<u8> = Vec::new();

    // Header
    buf.write_u32::<LittleEndian>(0x46554747).unwrap();
    buf.write_u32::<LittleEndian>(3).unwrap();
    buf.write_u64::<LittleEndian>(0).unwrap(); // no tensors
    buf.write_u64::<LittleEndian>(5).unwrap(); // 5 metadata entries

    // uint32
    write_string(&mut buf, "val_u32");
    buf.write_u32::<LittleEndian>(4).unwrap();
    buf.write_u32::<LittleEndian>(42).unwrap();

    // float32
    write_string(&mut buf, "val_f32");
    buf.write_u32::<LittleEndian>(6).unwrap();
    buf.write_f32::<LittleEndian>(3.14).unwrap();

    // bool (true)
    write_string(&mut buf, "val_bool");
    buf.write_u32::<LittleEndian>(7).unwrap();
    buf.write_u8(1).unwrap();

    // string
    write_string(&mut buf, "val_str");
    buf.write_u32::<LittleEndian>(8).unwrap();
    write_string(&mut buf, "hello");

    // array of uint32
    write_string(&mut buf, "val_arr");
    buf.write_u32::<LittleEndian>(9).unwrap(); // Array type
    buf.write_u32::<LittleEndian>(4).unwrap(); // element type: uint32
    buf.write_u64::<LittleEndian>(3).unwrap(); // count
    buf.write_u32::<LittleEndian>(10).unwrap();
    buf.write_u32::<LittleEndian>(20).unwrap();
    buf.write_u32::<LittleEndian>(30).unwrap();

    std::fs::write(path, &buf).unwrap();
}

// ---- Tests ----

#[test]
fn parse_valid_test_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_test_gguf(&path);

    let gguf = gguf_inspect::parse_gguf(&path).unwrap();

    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.metadata.len(), 3);
    assert_eq!(gguf.tensors.len(), 2);

    // Metadata checks
    assert_eq!(gguf.architecture(), Some("test"));
    assert_eq!(gguf.model_name(), Some("test-model"));
    assert_eq!(
        gguf.arch_metadata("block_count")
            .and_then(|v| v.as_u32()),
        Some(2)
    );

    // Tensor checks
    let t0 = &gguf.tensors[0];
    assert_eq!(t0.name, "weight_a");
    assert_eq!(t0.dimensions, vec![4, 8]);
    assert_eq!(t0.element_count(), 32);
    assert_eq!(t0.shape_str(), "4 x 8");

    let t1 = &gguf.tensors[1];
    assert_eq!(t1.name, "bias_b");
    assert_eq!(t1.dimensions, vec![8]);
    assert_eq!(t1.element_count(), 8);
}

#[test]
fn parse_total_parameters() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_test_gguf(&path);

    let gguf = gguf_inspect::parse_gguf(&path).unwrap();
    // weight_a: 4*8=32, bias_b: 8 => total 40
    assert_eq!(gguf.total_parameters(), 40);
}

#[test]
fn parse_total_tensor_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_test_gguf(&path);

    let gguf = gguf_inspect::parse_gguf(&path).unwrap();
    // weight_a: 32 elements * 32 bits / 8 = 128 bytes (F32)
    // bias_b: 8 elements * 16 bits / 8 = 16 bytes (F16)
    assert_eq!(gguf.total_tensor_size(), 128 + 16);
}

#[test]
fn reject_invalid_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_magic.gguf");
    create_invalid_magic_gguf(&path);

    let result = gguf_inspect::parse_gguf(&path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid GGUF magic"));
}

#[test]
fn reject_unsupported_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_version.gguf");
    create_bad_version_gguf(&path);

    let result = gguf_inspect::parse_gguf(&path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Unsupported GGUF version"));
}

#[test]
fn parse_multiple_value_types() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("multi.gguf");
    create_multi_type_gguf(&path);

    let gguf = gguf_inspect::parse_gguf(&path).unwrap();
    assert_eq!(gguf.metadata.len(), 5);

    // uint32
    assert_eq!(gguf.get_metadata("val_u32").and_then(|v| v.as_u32()), Some(42));

    // float32
    let f = gguf.get_metadata("val_f32");
    assert!(f.is_some());
    if let Some(gguf_inspect::GgufValue::Float32(v)) = f {
        assert!((v - 3.14).abs() < 0.001);
    } else {
        panic!("Expected Float32");
    }

    // bool
    if let Some(gguf_inspect::GgufValue::Bool(v)) = gguf.get_metadata("val_bool") {
        assert!(v);
    } else {
        panic!("Expected Bool(true)");
    }

    // string
    assert_eq!(
        gguf.get_metadata("val_str").and_then(|v| v.as_str()),
        Some("hello")
    );

    // array
    let arr = gguf.get_metadata("val_arr").and_then(|v| v.as_array());
    assert!(arr.is_some());
    let arr = arr.unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0].as_u32(), Some(10));
    assert_eq!(arr[1].as_u32(), Some(20));
    assert_eq!(arr[2].as_u32(), Some(30));
}

#[test]
fn parse_nonexistent_file() {
    let result = gguf_inspect::parse_gguf(Path::new("/tmp/nonexistent_gguf_file_12345.gguf"));
    assert!(result.is_err());
}

#[test]
fn cli_runs_on_test_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_test_gguf(&path);

    // Run the binary and verify it produces expected output
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .arg(path.to_str().unwrap())
        .output()
        .expect("failed to run gguf-inspect");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("GGUF Model Summary"));
    assert!(stdout.contains("Architecture:     test"));
    assert!(stdout.contains("Model name:       test-model"));
    assert!(stdout.contains("Tensor count:     2"));
}

#[test]
fn cli_json_output() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.gguf");
    create_test_gguf(&path);

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_gguf-inspect"))
        .arg(path.to_str().unwrap())
        .arg("--json")
        .output()
        .expect("failed to run gguf-inspect");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success());

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("invalid JSON output");
    assert_eq!(parsed["version"], 3);
    assert_eq!(parsed["tensor_count"], 2);
    assert_eq!(parsed["architecture"], "test");
}
