pub mod parser;
pub mod types;

// Re-export key types and the parse function for convenience.
pub use parser::parse_gguf;
pub use types::*;
