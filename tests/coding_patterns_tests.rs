mod common;

use async_openai::types::chat::{
    ChatCompletionRequestMessage, ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
};
use common::harness::{TestServerConfig, spawn_test_server};
use common::http_client::TestClient;

pub const RUST_PROMPT: &str = r#"\nuse std::collections::HashMap;\n\nfn main() {\n    let mut scores = HashMap::new();\n    scores.insert(\"Blue\", 10);\n    scores.insert(\"Yellow\", 50);\n\n    for (key, value) in &scores {\n        println!(\"{}: {}\", key, value);\n    }\n}\n"#;

pub const PYTHON_PROMPT: &str = r#"\nimport asyncio\n\nasync def fetch_data():\n    print(\"start fetching\")\n    await asyncio.sleep(2)\n    print(\"done fetching\")\n    return {\'data\': 1}\n\nasync def main():\n    result = await fetch_data()\n    print(result)\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n"#;

pub const JAVASCRIPT_PROMPT: &str = r#"\nconst express = require('express');\nconst app = express();\nconst port = 3000;\n\napp.get('/', (req, res) => {\n  res.send('Hello World!');\n});\n\napp.listen(port, () => {\n  console.log(`Example app listening on port ${port}`);\n});\n"#;

pub const SQL_PROMPT: &str = r#"\nSELECT
    u.id,
    u.username,
    COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at > '2023-01-01'
GROUP BY u.id
ORDER BY order_count DESC;\n"#;

pub const MARKDOWN_PROMPT: &str = r#"\n# API Documentation\n\n## Authentication\nAll API requests require an API key in the header.\n\n```bash\ncurl -H \"Authorization: Bearer <token>\" https://api.example.com/v1/users\n```\n\n## Response Format\nResponses are JSON formatted.\n"#;

pub const LOG_PROMPT: &str = r#"\n2023-10-27 10:00:00.123 INFO  [main] c.e.service.UserService - User 123 logged in\n2023-10-27 10:00:00.234 DEBUG [main] c.e.service.ProductService - Fetching product details for ID 456\n2023-10-27 10:00:00.567 WARN  [worker-1] c.e.job.BatchJob - Job processing took longer than expected: 1500ms\n2023-10-27 10:00:00.890 ERROR [worker-2] c.e.db.ConnectionPool - Failed to acquire connection\njava.net.ConnectException: Connection refused\n    at java.base/sun.nio.ch.Net.connect0(Native Method)\n    at java.base/sun.nio.ch.Net.connect(Net.java:579)\n"#;

pub const JSON_CONFIG_PROMPT: &str = r#"\n{\n  \"compilerOptions\": {\n    \"target\": \"es2016\",\n    \"module\": \"commonjs\",\n    \"strict\": true,\n    \"esModuleInterop\": true,\n    \"skipLibCheck\": true,\n    \"forceConsistentCasingInFileNames\": true\n  },\n  \"include\": [\"src/**/*\"],\n  \"exclude\": [\"node_modules\", \"**/*.spec.ts\"]\n}\n"#;

pub const HTML_PROMPT: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <div id="app"></div>
    <script src="main.js"></script>
</body>
</html>
"#;

pub const MIXED_CONTENT_PROMPT: &str = r#"
Here is an opening paragraph.

{
  "key": "value",
  "list": [1, 2, 3]
}

And here is a table:
| ID | Name |
|----|------|
| 1  | Rust |
| 2  | Tauq |
"#;

pub const ALL_CODING_PROMPTS: &[&str] = &[
    RUST_PROMPT,
    PYTHON_PROMPT,
    JAVASCRIPT_PROMPT,
    SQL_PROMPT,
    MARKDOWN_PROMPT,
    LOG_PROMPT,
    JSON_CONFIG_PROMPT,
    HTML_PROMPT,
    MIXED_CONTENT_PROMPT,
];

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
async fn test_coding_patterns_persistence() {
    let server = spawn_test_server(TestServerConfig::default())
        .await
        .expect("Server should start");

    let client = TestClient::new(server.url());

    for (i, prompt) in ALL_CODING_PROMPTS.iter().enumerate() {
        println!("Testing prompt {}: {:.20}...", i, prompt.replace("\n", " "));

        let request = create_request(prompt);
        let (response, status) = client
            .chat_completions(request)
            .await
            .expect("Request should succeed");
        assert_eq!(
            status, "MISS",
            "Expected MISS for new prompt: {:.20}...",
            prompt
        );
        assert!(!response.id.is_empty());

        let request_cached = create_request(prompt);
        let (response_cached, status_cached) = client
            .chat_completions(request_cached)
            .await
            .expect("Request should succeed");
        assert!(
            status_cached.starts_with("HIT"),
            "Expected HIT status (got {}) for cached prompt: {:.20}...",
            status_cached,
            prompt
        );
        assert_eq!(response.id, response_cached.id, "Response ID should match");
    }
}
