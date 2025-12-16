//! End-to-end HTTP tests.

mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use reflex::REFLEX_STATUS_READY;

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
async fn test_health_endpoint_returns_ok() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());
    let health = client.health().await.expect("Health check should succeed");

    assert_eq!(health.status, "ok");
}

#[tokio::test]
async fn test_ready_endpoint_indicates_dependencies() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());
    let ready = client.ready().await.expect("Ready check should succeed");

    assert!(ready.is_ok(), "Server should report ready");
    assert_eq!(ready.components.http, REFLEX_STATUS_READY);
}

#[tokio::test]
async fn test_health_endpoint_responds_quickly() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());

    let start = std::time::Instant::now();
    let _ = client.health().await;
    let elapsed = start.elapsed();

    assert!(
        elapsed.as_millis() < 100,
        "Health check took {}ms, expected < 100ms",
        elapsed.as_millis()
    );
}

#[tokio::test]
async fn test_chat_completion_basic() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());
    let request = create_request("Hello E2E");

    let (response, status) = client
        .chat_completions(request)
        .await
        .expect("Request should succeed");

    assert_eq!(status, "MISS");
    assert!(!response.id.is_empty());
}

#[tokio::test]
async fn test_concurrent_requests() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let client = TestClient::new(server.url());
            tokio::spawn(async move {
                let request = create_request(&format!("Concurrent content {}", i));
                client.chat_completions(request).await
            })
        })
        .collect();

    let results = futures::future::join_all(handles).await;

    for (i, result) in results.into_iter().enumerate() {
        let inner: Result<_, _> = result.expect("Task should not panic");
        assert!(inner.is_ok(), "Request {} should succeed", i);
    }
}

#[tokio::test]
async fn test_server_lifecycle() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());
    let health = client.health().await;
    assert!(health.is_ok(), "Server should be healthy after startup");

    server.shutdown().await;

    let result = client.health().await;
    assert!(
        result.is_err(),
        "Server should reject connections after shutdown"
    );
}

#[tokio::test]
async fn test_server_handles_dropped_connections() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());

    client.health().await.expect("First request should succeed");

    for _ in 0..100 {
        let _ = client.health().await;
    }

    client
        .health()
        .await
        .expect("Server should handle connection churn");
}
