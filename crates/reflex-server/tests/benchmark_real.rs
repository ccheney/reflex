mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use common::harness::{TestServerConfig, spawn_real_server};
use common::http_client::TestClient;
use std::time::Instant;

fn create_request(prompt: &str) -> async_openai::types::chat::CreateChatCompletionRequest {
    CreateChatCompletionRequestArgs::default()
        .model("gpt-4o")
        .messages([ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessageArgs::default()
                .content(prompt)
                .build()
                .unwrap(),
        )])
        .build()
        .unwrap()
}

#[tokio::test]
#[ignore]
async fn test_benchmark_performance() {
    println!("\n=== Starting Performance Benchmark ===");

    let cwd = std::env::current_dir().expect("Failed to get CWD");

    let model_path = cwd.join(".models/qwen2.5-0.5b-instruct-q8_0.gguf");
    if model_path.exists() {
        println!("> Using Local Embedder: {:?}", model_path);
        unsafe {
            std::env::set_var("REFLEX_MODEL_PATH", model_path);
        }
    } else {
        println!("> Local Embedder not found. Using Stub Embedder.");
    }

    let reranker_path = cwd.join(".models/ms-marco-MiniLM-L-6-v2");
    if reranker_path.exists() {
        println!("> Using Local Reranker: {:?}", reranker_path);
        unsafe {
            std::env::set_var("REFLEX_RERANKER_PATH", reranker_path);
        }
    } else {
        println!("> Local Reranker not found. Downloading via hf-hub...");
        println!("> Skipping download complexity. Will run with Stub Reranker.");
    }

    let config = TestServerConfig::default();
    let server = spawn_real_server(config)
        .await
        .expect("Failed to spawn server");
    let client = TestClient::new(server.url());

    let prompt = "How do I implement a binary search tree in Rust with interior mutability?";
    let req = create_request(prompt);

    println!("> Sending Initial Request (Miss)...");
    let start_miss = Instant::now();
    let (_, status1) = client
        .chat_completions(req.clone())
        .await
        .expect("Request failed");
    let duration_miss = start_miss.elapsed();
    println!("  Status: {}", status1);
    println!("  Latency: {:.2?}", duration_miss);

    println!("> Waiting for Async Indexing...");
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    println!("> Sending Second Request (Targeting Hit)...");
    let prompt2 = "Show me a Rust binary search tree implementation using interior mutability.";
    let req2 = create_request(prompt2);
    let start_hit = Instant::now();
    let (_, status2) = client.chat_completions(req2).await.expect("Request failed");
    let duration_hit = start_hit.elapsed();
    println!("  Status: {}", status2);
    println!("  Latency: {:.2?}", duration_hit);

    println!("\n--- Results ---");
    println!("Miss Latency: {:.2?}", duration_miss);
    println!("Hit Latency:  {:.2?}", duration_hit);
    if duration_hit < duration_miss {
        let speedup = duration_miss.as_secs_f64() / duration_hit.as_secs_f64();
        println!("Speedup: {:.2}x", speedup);
    } else {
        println!("Speedup: None (Hit was slower?)");
    }
    println!("======================================\n");
}
