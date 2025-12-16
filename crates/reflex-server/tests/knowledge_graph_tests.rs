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
async fn test_knowledge_graph_semantic_retrieval() {
    let rust_store = "Explain the difference between Arc and Mutex in Rust programming language. Arc (Atomic Reference Counted) provides shared ownership of data across multiple threads by keeping a reference count in atomic memory. Mutex (Mutual Exclusion) ensures exclusive mutable access to the underlying data using locking mechanisms to prevent data races. They are often used together as Arc<Mutex<T>>.";
    let rust_query = "What is the difference between Arc and Mutex in Rust programming language for shared ownership and threads safety? I need to know about Atomic Reference Counted and exclusive mutable access using locking mechanisms.";

    let space_store = "The Apollo 11 mission, launched by NASA, successfully landed astronauts Neil Armstrong and Buzz Aldrin on the Moon's surface in the Mare Tranquillitatis (Sea of Tranquility) region on July 20, 1969. This historic event marked the first time humans set foot on another planetary body, fulfilling President John F. Kennedy's national goal.";
    let space_query = "When did the NASA Apollo 11 mission land on the Moon surface? Who were the astronauts Neil Armstrong and Buzz Aldrin that walked on the Sea of Tranquility in 1969?";

    let food_store = "To achieve the perfect Maillard reaction when searing a steak, ensure the meat is pat dry to remove moisture and the cast iron pan is extremely hot. This chemical reaction between amino acids and reducing sugars creates the distinctive browned crust and complex savory flavors that characterize a well-cooked steak.";
    let food_query = "How do I get a Maillard reaction crust when searing steak using a hot cast iron pan? I want to ensure the meat is pat dry to remove moisture and create complex savory flavors.";

    let server = spawn_test_server(TestServerConfig::default())
        .await
        .unwrap();
    let client = TestClient::new(server.url());

    client
        .chat_completions(create_request(rust_store))
        .await
        .expect("Store Rust");
    client
        .chat_completions(create_request(space_store))
        .await
        .expect("Store Space");
    client
        .chat_completions(create_request(food_store))
        .await
        .expect("Store Food");

    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    let (resp_rust, status_rust) = client
        .chat_completions(create_request(rust_query))
        .await
        .unwrap();
    println!("Rust Status: {}", status_rust);
    let content_rust = resp_rust.choices[0].message.content.as_ref().unwrap();
    assert!(
        content_rust.contains("Atomic Reference Counted"),
        "Expected Rust content"
    );

    let (resp_space, status_space) = client
        .chat_completions(create_request(space_query))
        .await
        .unwrap();
    println!("Space Status: {}", status_space);
    let content_space = resp_space.choices[0].message.content.as_ref().unwrap();
    assert!(
        content_space.contains("Apollo 11"),
        "Expected Space content"
    );

    let (resp_food, status_food) = client
        .chat_completions(create_request(food_query))
        .await
        .unwrap();
    println!("Food Status: {}", status_food);
    let content_food = resp_food.choices[0].message.content.as_ref().unwrap();
    assert!(content_food.contains("Maillard"), "Expected Food content");

    let biology_query =
        "Explain the process of photosynthesis in plants converting light into energy.";
    let (_resp_bio, status_bio) = client
        .chat_completions(create_request(biology_query))
        .await
        .unwrap();
    println!("Biology Status: {}", status_bio);
    assert_eq!(status_bio, "MISS", "Should miss for unrelated content");
}
