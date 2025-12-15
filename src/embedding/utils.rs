use std::io;
use std::path::Path;
use tokenizers::Tokenizer;

/// Loads a tokenizer from a model directory or explicit tokenizer.json path.
pub fn load_tokenizer(model_path: &Path) -> io::Result<Tokenizer> {
    let tokenizer_path = if model_path
        .file_name()
        .is_some_and(|name| name == std::ffi::OsStr::new("tokenizer.json"))
    {
        model_path.to_path_buf()
    } else if model_path.is_dir() {
        model_path.join("tokenizer.json")
    } else {
        model_path
            .parent()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Model path has no parent"))?
            .join("tokenizer.json")
    };

    Tokenizer::from_file(&tokenizer_path).map_err(io::Error::other)
}

/// Loads a tokenizer with truncation enabled for a maximum sequence length.
///
/// This is important for cross-encoder models that have a fixed maximum sequence length.
/// Inputs exceeding `max_len` will be truncated to fit.
pub fn load_tokenizer_with_truncation(model_path: &Path, max_len: usize) -> io::Result<Tokenizer> {
    use tokenizers::TruncationParams;

    let mut tokenizer = load_tokenizer(model_path)?;

    let truncation = TruncationParams {
        max_length: max_len,
        ..Default::default()
    };

    tokenizer
        .with_truncation(Some(truncation))
        .map_err(|e| io::Error::other(format!("Failed to configure truncation: {}", e)))?;

    Ok(tokenizer)
}
