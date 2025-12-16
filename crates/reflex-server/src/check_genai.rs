use genai::chat::{ChatMessage, ChatRequest};
use genai::Client;

async fn check() {
    let client = Client::default();
    
    let req = ChatRequest::new(vec![
        ChatMessage::user("Hello"),
    ]);
}

fn main() {}
