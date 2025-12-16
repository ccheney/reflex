mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
};
use common::harness::{TestServerConfig, spawn_test_server};
use common::http_client::TestClient;

fn create_request(
    model: &str,
    prompt: &str,
) -> async_openai::types::chat::CreateChatCompletionRequest {
    CreateChatCompletionRequestArgs::default()
        .model(model)
        .messages([
            ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content("You are a helpful assistant.")
                    .build()
                    .unwrap(),
            ),
            ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content(prompt)
                    .build()
                    .unwrap(),
            ),
        ])
        .build()
        .unwrap()
}

#[tokio::test]
async fn test_openai_completion_lifecycle() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .unwrap();
    let client = TestClient::new(server.url());

    let request = create_request("gpt-4o", "Hello world from OpenAI test");

    let (resp1, status1) = client.chat_completions(request.clone()).await.unwrap();
    assert_eq!(status1, "MISS");
    assert_eq!(resp1.model, "gpt-4o");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let (resp2, status2) = client.chat_completions(request.clone()).await.unwrap();
    assert_eq!(status2, "HIT_L1_EXACT");
    assert_eq!(resp2.model, "gpt-4o");

    assert!(
        resp2.choices[0]
            .message
            .content
            .as_ref()
            .unwrap()
            .contains("Mock response")
    );
}

#[tokio::test]
async fn test_claude_via_openai_format_lifecycle() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .unwrap();
    let client = TestClient::new(server.url());

    let request = create_request("claude-3-opus", "Hello world from Claude test");

    let (resp1, status1) = client.chat_completions(request.clone()).await.unwrap();
    assert_eq!(status1, "MISS");
    assert_eq!(resp1.model, "claude-3-opus");

    tokio::time::sleep(std::time::Duration::from_millis(500)).await;

    let (resp2, status2) = client.chat_completions(request.clone()).await.unwrap();
    assert_eq!(status2, "HIT_L1_EXACT");
    assert_eq!(resp2.model, "claude-3-opus");
}
