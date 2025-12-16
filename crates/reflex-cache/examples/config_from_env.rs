//! Load configuration from environment variables.

use anyhow::Result;

fn main() -> Result<()> {
    let config = reflex::Config::from_env()?;
    println!(
        "bind_addr={}, port={}, qdrant_url={}",
        config.bind_addr, config.port, config.qdrant_url
    );
    Ok(())
}
