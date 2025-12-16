//! Create a stub embedder and generate an embedding.

use anyhow::Result;

fn main() -> Result<()> {
    use reflex::{SinterConfig, SinterEmbedder};

    let embedder = SinterEmbedder::load(SinterConfig::stub())?;
    let embedding = embedder.embed("hello world")?;
    println!("dim={}", embedding.len());
    Ok(())
}
