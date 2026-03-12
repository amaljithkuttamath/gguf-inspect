use serde::Serialize;
use std::fmt;

/// GGUF magic number: "GGUF" in little-endian.
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Metadata value types in GGUF format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Uint8),
            1 => Some(Self::Int8),
            2 => Some(Self::Uint16),
            3 => Some(Self::Int16),
            4 => Some(Self::Uint32),
            5 => Some(Self::Int32),
            6 => Some(Self::Float32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::Uint64),
            11 => Some(Self::Int64),
            12 => Some(Self::Float64),
            _ => None,
        }
    }
}

/// A parsed metadata value.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl fmt::Display for GgufValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uint8(v) => write!(f, "{v}"),
            Self::Int8(v) => write!(f, "{v}"),
            Self::Uint16(v) => write!(f, "{v}"),
            Self::Int16(v) => write!(f, "{v}"),
            Self::Uint32(v) => write!(f, "{v}"),
            Self::Int32(v) => write!(f, "{v}"),
            Self::Float32(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "{v}"),
            Self::Uint64(v) => write!(f, "{v}"),
            Self::Int64(v) => write!(f, "{v}"),
            Self::Float64(v) => write!(f, "{v}"),
            Self::Array(arr) => {
                if arr.len() <= 5 {
                    let items: Vec<String> = arr.iter().map(|v| format!("{v}")).collect();
                    write!(f, "[{}]", items.join(", "))
                } else {
                    let first: Vec<String> = arr[..3].iter().map(|v| format!("{v}")).collect();
                    write!(f, "[{}, ... ({} total)]", first.join(", "), arr.len())
                }
            }
        }
    }
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// GGML tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            other => Self::Unknown(other),
        }
    }

    /// Bits per weight for this quantization type.
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            Self::F32 => 32.0,
            Self::F16 => 16.0,
            Self::Q4_0 => 4.5,
            Self::Q4_1 => 5.0,
            Self::Q5_0 => 5.5,
            Self::Q5_1 => 6.0,
            Self::Q8_0 => 8.5,
            Self::Q8_1 => 9.0,
            Self::Q2K => 2.5625,
            Self::Q3K => 3.4375,
            Self::Q4K => 4.5,
            Self::Q5K => 5.5,
            Self::Q6K => 6.5625,
            Self::Q8K => 8.5,
            Self::IQ2XXS => 2.06,
            Self::IQ2XS => 2.31,
            Self::IQ3XXS => 3.06,
            Self::Unknown(_) => 4.0, // reasonable default
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "F32"),
            Self::F16 => write!(f, "F16"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q5_0 => write!(f, "Q5_0"),
            Self::Q5_1 => write!(f, "Q5_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q2K => write!(f, "Q2_K"),
            Self::Q3K => write!(f, "Q3_K"),
            Self::Q4K => write!(f, "Q4_K"),
            Self::Q5K => write!(f, "Q5_K"),
            Self::Q6K => write!(f, "Q6_K"),
            Self::Q8K => write!(f, "Q8_K"),
            Self::IQ2XXS => write!(f, "IQ2_XXS"),
            Self::IQ2XS => write!(f, "IQ2_XS"),
            Self::IQ3XXS => write!(f, "IQ3_XXS"),
            Self::Unknown(v) => write!(f, "unknown({v})"),
        }
    }
}

/// File type enum for the general.file_type metadata key.
pub fn file_type_name(ft: u32) -> &'static str {
    match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q5_0",
        8 => "Q5_1",
        9 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "IQ3_XXS",
        _ => "unknown",
    }
}

/// Parsed tensor info entry.
#[derive(Debug, Clone, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements in this tensor.
    pub fn element_count(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 0;
        }
        self.dimensions.iter().product()
    }

    /// Estimated size in bytes.
    pub fn size_bytes(&self) -> u64 {
        let elements = self.element_count() as f64;
        (elements * self.dtype.bits_per_weight() / 8.0) as u64
    }

    /// Shape as a human-readable string.
    pub fn shape_str(&self) -> String {
        let dims: Vec<String> = self.dimensions.iter().map(|d| d.to_string()).collect();
        dims.join(" x ")
    }
}

/// Full parsed GGUF file.
#[derive(Debug, Clone, Serialize)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
    pub file_size: u64,
}

impl GgufFile {
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.get_metadata("general.architecture")?.as_str()
    }

    pub fn model_name(&self) -> Option<&str> {
        self.get_metadata("general.name")?.as_str()
    }

    /// Look up an architecture-specific key like "{arch}.context_length".
    pub fn arch_metadata(&self, suffix: &str) -> Option<&GgufValue> {
        let arch = self.architecture()?;
        self.get_metadata(&format!("{arch}.{suffix}"))
    }

    pub fn total_parameters(&self) -> u64 {
        self.tensors.iter().map(|t| t.element_count()).sum()
    }

    pub fn total_tensor_size(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }
}
