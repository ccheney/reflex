mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use common::harness::{TestServerConfig, spawn_test_server};
use common::http_client::TestClient;

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
async fn test_reranker_selects_best_candidate() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .unwrap();
    let client = TestClient::new(server.url());

    let prompt_a = "How to implement a binary tree in Rust. This includes struct definitions, insert methods, and traversal logic for a complete implementation.";
    client
        .chat_completions(create_request(prompt_a))
        .await
        .expect("Request A failed");

    let prompt_b = "How to implement a binary tree in Java. This includes class definitions, insert methods, and traversal logic for a complete implementation.";
    client
        .chat_completions(create_request(prompt_b))
        .await
        .expect("Request B failed");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let query = "How do I implement a binary tree in Rust? I need struct definitions, insert methods, and traversal logic.";
    let (response, status) = client
        .chat_completions(create_request(query))
        .await
        .expect("Query failed");

    assert_eq!(status, "HIT_L3_VERIFIED");

    let content = response.choices[0].message.content.as_ref().unwrap();
    assert!(
        content.contains("Rust."),
        "Expected match for Rust prompt, got: {}",
        content
    );
    assert!(!content.contains("Java."), "Should not match Java prompt");
}

#[tokio::test]
async fn test_reranker_rejects_poor_matches() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .unwrap();
    let client = TestClient::new(server.url());

    let prompt = "The weather in Paris is nice today.";
    client
        .chat_completions(create_request(prompt))
        .await
        .unwrap();

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let query = "How to implement a binary tree in Rust?";

    let (_, status) = client
        .chat_completions(create_request(query))
        .await
        .unwrap();

    assert_eq!(status, "MISS", "Expected Miss due to low reranker score");
}
