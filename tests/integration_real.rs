mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use common::harness::{TestServerConfig, spawn_real_server};
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
async fn test_real_health_check() {
    let server = spawn_real_server(TestServerConfig::default())
        .await
        .expect("Failed to spawn real server");

    let client = TestClient::new(server.url());
    let health = client.health().await.expect("Health check failed");
    assert_eq!(health.status, "ok");
}

#[tokio::test]
async fn test_real_chat_lifecycle() {
    let server = spawn_real_server(TestServerConfig::default())
        .await
        .expect("Failed to spawn real server");

    let client = TestClient::new(server.url());

    let req = create_request("The capital of France is Paris.");

    let (_resp1, status1) = client
        .chat_completions(req.clone())
        .await
        .expect("First request failed");
    assert_eq!(status1, reflex::ReflexStatus::Miss.as_header_value());

    let (_resp2, status2) = client
        .chat_completions(req.clone())
        .await
        .expect("Second request failed");
    assert_eq!(status2, reflex::ReflexStatus::HitL1Exact.as_header_value());
}
