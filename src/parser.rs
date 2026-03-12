use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

use crate::types::*;

/// Parse a GGUF file from the given path.
pub fn parse_gguf(path: &Path) -> io::Result<GgufFile> {
    let file_size = path.metadata()?.len();
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Magic number
    let magic = reader.read_u32::<LittleEndian>()?;
    if magic != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Invalid GGUF magic: expected 0x{GGUF_MAGIC:08X}, got 0x{magic:08X}"),
        ));
    }

    // Version
    let version = reader.read_u32::<LittleEndian>()?;
    if !(2..=3).contains(&version) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version: {version}"),
        ));
    }

    // Counts
    let tensor_count = reader.read_u64::<LittleEndian>()?;
    let metadata_kv_count = reader.read_u64::<LittleEndian>()?;

    // Metadata
    let mut metadata = Vec::with_capacity(metadata_kv_count as usize);
    for _ in 0..metadata_kv_count {
        let key = read_string(&mut reader)?;
        let value_type = reader.read_u32::<LittleEndian>()?;
        let value = read_value(&mut reader, value_type)?;
        metadata.push((key, value));
    }

    // Tensor info
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(reader.read_u64::<LittleEndian>()?);
        }
        let dtype_raw = reader.read_u32::<LittleEndian>()?;
        let dtype = GgmlType::from_u32(dtype_raw);
        let offset = reader.read_u64::<LittleEndian>()?;
        tensors.push(TensorInfo {
            name,
            dimensions,
            dtype,
            offset,
        });
    }

    Ok(GgufFile {
        version,
        metadata,
        tensors,
        file_size,
    })
}

fn read_string(reader: &mut BufReader<File>) -> io::Result<String> {
    let len = reader.read_u64::<LittleEndian>()? as usize;
    if len > 10_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("String length too large: {len}"),
        ));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_value(reader: &mut BufReader<File>, value_type: u32) -> io::Result<GgufValue> {
    let vtype = GgufValueType::from_u32(value_type).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown value type: {value_type}"),
        )
    })?;

    match vtype {
        GgufValueType::Uint8 => Ok(GgufValue::Uint8(reader.read_u8()?)),
        GgufValueType::Int8 => Ok(GgufValue::Int8(reader.read_i8()?)),
        GgufValueType::Uint16 => Ok(GgufValue::Uint16(reader.read_u16::<LittleEndian>()?)),
        GgufValueType::Int16 => Ok(GgufValue::Int16(reader.read_i16::<LittleEndian>()?)),
        GgufValueType::Uint32 => Ok(GgufValue::Uint32(reader.read_u32::<LittleEndian>()?)),
        GgufValueType::Int32 => Ok(GgufValue::Int32(reader.read_i32::<LittleEndian>()?)),
        GgufValueType::Float32 => Ok(GgufValue::Float32(reader.read_f32::<LittleEndian>()?)),
        GgufValueType::Bool => Ok(GgufValue::Bool(reader.read_u8()? != 0)),
        GgufValueType::String => Ok(GgufValue::String(read_string(reader)?)),
        GgufValueType::Uint64 => Ok(GgufValue::Uint64(reader.read_u64::<LittleEndian>()?)),
        GgufValueType::Int64 => Ok(GgufValue::Int64(reader.read_i64::<LittleEndian>()?)),
        GgufValueType::Float64 => Ok(GgufValue::Float64(reader.read_f64::<LittleEndian>()?)),
        GgufValueType::Array => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let count = reader.read_u64::<LittleEndian>()? as usize;
            let mut items = Vec::with_capacity(count.min(1_000_000));
            for _ in 0..count {
                items.push(read_value(reader, elem_type)?);
            }
            Ok(GgufValue::Array(items))
        }
    }
}
