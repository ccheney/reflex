//! Basic tiered-cache lookup flow.

use anyhow::Result;

#[cfg(feature = "mock")]
#[tokio::main]
async fn main() -> Result<()> {
    use reflex::{MockTieredCache, TieredLookupResult};

    let cache = MockTieredCache::new_mock().await?;
    let tenant_id = 1;

    match cache.lookup("hello", tenant_id).await? {
        TieredLookupResult::HitL1(_) => println!("L1 hit"),
        TieredLookupResult::HitL2(r) => println!("L2 hit ({} candidates)", r.candidates().len()),
        TieredLookupResult::Miss => println!("miss"),
    }

    Ok(())
}

#[cfg(not(feature = "mock"))]
fn main() {
    eprintln!("Run with: cargo run --example basic_lookup --features mock");
}

