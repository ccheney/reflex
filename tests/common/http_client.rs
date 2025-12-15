//! HTTP client helpers for tests.

use async_openai::types::chat::{CreateChatCompletionRequest, CreateChatCompletionResponse};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 10;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(DEFAULT_TIMEOUT_SECS);

pub struct TestClient {
    client: reqwest::Client,
    base_url: String,
}

impl TestClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            base_url: base_url.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        let path = path.trim_start_matches('/');
        format!("{}/{}", self.base_url, path)
    }

    fn add_headers(&self, builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        builder.header("Content-Type", "application/json")
    }

    pub async fn chat_completions(
        &self,
        request: CreateChatCompletionRequest,
    ) -> Result<(CreateChatCompletionResponse, String), TestClientError> {
        let builder = self.client.post(self.url("/v1/chat/completions"));
        let builder = self.add_headers(builder);

        let resp = builder.json(&request).send().await?;

        let status_header = resp
            .headers()
            .get("x-reflex-status")
            .and_then(|h| h.to_str().ok())
            .unwrap_or("unknown")
            .to_string();

        match resp.status().as_u16() {
            200 => Ok((resp.json().await?, status_header)),
            400 | 422 => Err(TestClientError::BadRequest(resp.text().await?)),
            status => {
                let body = resp.text().await.unwrap_or_default();
                Err(TestClientError::UnexpectedStatus(status, body))
            }
        }
    }

    pub async fn health(&self) -> Result<HealthResponse, TestClientError> {
        let resp = self.client.get(self.url("/healthz")).send().await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            Err(TestClientError::UnexpectedStatus(status, body))
        }
    }

    pub async fn ready(&self) -> Result<ReadyResponse, TestClientError> {
        let resp = self.client.get(self.url("/ready")).send().await?;

        if resp.status().is_success() {
            Ok(resp.json().await?)
        } else {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            Err(TestClientError::UnexpectedStatus(status, body))
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HealthResponse {
    pub status: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComponentStatus {
    pub http: String,
    pub storage: String,
    pub vectordb: String,
    pub embedding: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReadyResponse {
    pub status: String,
    pub components: ComponentStatus,
}

impl ReadyResponse {
    pub fn is_ok(&self) -> bool {
        self.status == "ok"
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TestClientError {
    #[error("HTTP request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),

    #[error("Unexpected HTTP status: {0} - Body: {1}")]
    UnexpectedStatus(u16, String),

    #[error("Bad request: {0}")]
    BadRequest(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_openai::types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    };

    #[test]
    fn test_client_url_building() {
        let client = TestClient::new("http://localhost:8080");
        assert_eq!(client.url("/healthz"), "http://localhost:8080/healthz");
        assert_eq!(client.url("healthz"), "http://localhost:8080/healthz");
    }

    #[test]
    fn test_client_exposes_endpoints() {
        let client = TestClient::new("http://localhost:8080");
        std::mem::drop(client.health());
        std::mem::drop(client.ready());

        let req = CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages([ChatCompletionRequestMessage::User(
                ChatCompletionRequestUserMessageArgs::default()
                    .content("hello")
                    .build()
                    .unwrap(),
            )])
            .build()
            .unwrap();
        std::mem::drop(client.chat_completions(req));
    }

    #[test]
    fn test_ready_response_is_ok_helper() {
        let ready = ReadyResponse {
            status: "ok".to_string(),
            components: ComponentStatus {
                http: "ready".to_string(),
                storage: "ready".to_string(),
                vectordb: "ready".to_string(),
                embedding: "ready".to_string(),
            },
        };
        assert!(ready.is_ok());
    }
}
