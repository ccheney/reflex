use async_openai::types::chat::{
    ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
    ChatCompletionRequestAssistantMessageContent, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageContent,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessageContent,
    ChatCompletionRequestUserMessageContentPart, ChatCompletionResponseMessage,
    ChatCompletionToolChoiceOption, ChatCompletionTools, CompletionUsage,
    CreateChatCompletionRequest, CreateChatCompletionResponse, FinishReason, FunctionCall,
    ToolChoiceOptions,
};
use genai::chat::{
    ChatMessage, ChatRequest, ChatResponse, MessageContent, Tool, ToolCall, ToolResponse,
};
use serde_json::Value;

pub fn adapt_openai_to_genai(req: CreateChatCompletionRequest) -> ChatRequest {
    let messages: Vec<ChatMessage> = req
        .messages
        .iter()
        .filter_map(|m| openai_message_to_genai_message(m.clone()))
        .collect();

    let mut chat_req = ChatRequest::new(messages);

    if let Some(tools) = &req.tools {
        let genai_tools: Vec<Tool> = tools.iter().filter_map(openai_tool_to_genai_tool).collect();
        if !genai_tools.is_empty() {
            chat_req = chat_req.with_tools(genai_tools);
        }
    }

    if let Some(tool_choice) = &req.tool_choice {
        match tool_choice {
            ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None) => {}
            ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto) => {}
            ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Required) => {}
            ChatCompletionToolChoiceOption::Function(named) => {
                tracing::debug!(
                    "tool_choice specifies function '{}', but genai does not support forced tool selection",
                    named.function.name
                );
            }
            _ => {}
        }
    }

    chat_req
}

fn openai_tool_to_genai_tool(tool: &ChatCompletionTools) -> Option<Tool> {
    match tool {
        ChatCompletionTools::Function(func_tool) => {
            let func = &func_tool.function;
            let mut genai_tool = Tool::new(&func.name);

            if let Some(desc) = &func.description {
                genai_tool = genai_tool.with_description(desc);
            }

            if let Some(params) = &func.parameters {
                genai_tool = genai_tool.with_schema(params.clone());
            }

            Some(genai_tool)
        }
        ChatCompletionTools::Custom(_) => None,
    }
}

pub fn adapt_genai_to_openai(resp: ChatResponse, model: String) -> CreateChatCompletionResponse {
    let tool_calls = resp.tool_calls();
    let content = resp.first_text().unwrap_or_default().to_string();

    let openai_tool_calls: Vec<ChatCompletionMessageToolCalls> = tool_calls
        .into_iter()
        .map(|tc| {
            ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                id: tc.call_id.clone(),
                function: FunctionCall {
                    name: tc.fn_name.clone(),
                    arguments: serde_json::to_string(&tc.fn_arguments)
                        .unwrap_or_else(|_| "{}".to_string()),
                },
            })
        })
        .collect();

    let message_value = serde_json::json!({
        "role": "assistant",
        "content": if content.trim().is_empty() { serde_json::Value::Null } else { serde_json::Value::String(content) },
        "tool_calls": if openai_tool_calls.is_empty() { serde_json::Value::Null } else { serde_json::to_value(openai_tool_calls).unwrap_or(serde_json::Value::Null) },
    });

    let message: ChatCompletionResponseMessage =
        serde_json::from_value(message_value).expect("constructed OpenAI message is valid");

    let response_value = serde_json::json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp() as u32,
        "model": model,
        "choices": vec![ChatChoice {
            index: 0,
            message,
            finish_reason: Some(FinishReason::Stop),
            logprobs: None,
        }],
        "usage": Some(CompletionUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        }),
    });

    serde_json::from_value(response_value).expect("constructed OpenAI response is valid")
}

fn openai_message_to_genai_message(m: ChatCompletionRequestMessage) -> Option<ChatMessage> {
    match m {
        ChatCompletionRequestMessage::Developer(dev) => Some(ChatMessage::system(
            openai_developer_content_to_text(dev.content),
        )),
        ChatCompletionRequestMessage::System(sys) => Some(ChatMessage::system(
            openai_system_content_to_text(sys.content),
        )),
        ChatCompletionRequestMessage::User(user) => {
            Some(ChatMessage::user(openai_user_content_to_text(user.content)))
        }
        ChatCompletionRequestMessage::Assistant(asst) => {
            let mut content = MessageContent::default();

            if let Some(tool_calls) = asst.tool_calls {
                for tc in tool_calls {
                    match tc {
                        ChatCompletionMessageToolCalls::Function(tc) => {
                            let args: Value = serde_json::from_str(&tc.function.arguments)
                                .unwrap_or_else(|_| Value::String(tc.function.arguments));
                            content.push(genai::chat::ContentPart::ToolCall(ToolCall {
                                call_id: tc.id,
                                fn_name: tc.function.name,
                                fn_arguments: args,
                            }));
                        }
                        ChatCompletionMessageToolCalls::Custom(tc) => {
                            content.push(genai::chat::ContentPart::ToolCall(ToolCall {
                                call_id: tc.id,
                                fn_name: tc.custom_tool.name,
                                fn_arguments: serde_json::json!({ "input": tc.custom_tool.input }),
                            }));
                        }
                    }
                }
            }

            if let Some(asst_content) = asst.content {
                let text = openai_assistant_content_to_text(asst_content);
                if !text.trim().is_empty() {
                    content.push(genai::chat::ContentPart::Text(text));
                }
            }

            if let Some(refusal) = asst.refusal
                && !refusal.trim().is_empty()
            {
                content.push(genai::chat::ContentPart::Text(refusal));
            }

            if content.is_empty() {
                return None;
            }

            Some(ChatMessage::assistant(content))
        }
        ChatCompletionRequestMessage::Tool(tool) => Some(ChatMessage::from(ToolResponse::new(
            tool.tool_call_id,
            openai_tool_content_to_text(tool.content),
        ))),
        ChatCompletionRequestMessage::Function(_) => None,
    }
}

fn openai_developer_content_to_text(
    content: ChatCompletionRequestDeveloperMessageContent,
) -> String {
    match content {
        ChatCompletionRequestDeveloperMessageContent::Text(t) => t,
        ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
            .into_iter()
            .map(|p| {
                match p {
                async_openai::types::chat::ChatCompletionRequestDeveloperMessageContentPart::Text(
                    t,
                ) => t.text,
            }
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn openai_system_content_to_text(content: ChatCompletionRequestSystemMessageContent) -> String {
    match content {
        ChatCompletionRequestSystemMessageContent::Text(t) => t,
        ChatCompletionRequestSystemMessageContent::Array(parts) => parts
            .into_iter()
            .map(|p| match p {
                async_openai::types::chat::ChatCompletionRequestSystemMessageContentPart::Text(
                    t,
                ) => t.text,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn openai_assistant_content_to_text(
    content: ChatCompletionRequestAssistantMessageContent,
) -> String {
    match content {
        ChatCompletionRequestAssistantMessageContent::Text(t) => t,
        ChatCompletionRequestAssistantMessageContent::Array(parts) => parts
            .into_iter()
            .map(|p| match p {
                async_openai::types::chat::ChatCompletionRequestAssistantMessageContentPart::Text(
                    t,
                ) => t.text,
                async_openai::types::chat::ChatCompletionRequestAssistantMessageContentPart::Refusal(
                    r,
                ) => r.refusal,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn openai_tool_content_to_text(content: ChatCompletionRequestToolMessageContent) -> String {
    match content {
        ChatCompletionRequestToolMessageContent::Text(t) => t,
        ChatCompletionRequestToolMessageContent::Array(parts) => parts
            .into_iter()
            .map(|p| match p {
                async_openai::types::chat::ChatCompletionRequestToolMessageContentPart::Text(t) => {
                    t.text
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn openai_user_content_to_text(content: ChatCompletionRequestUserMessageContent) -> String {
    match content {
        ChatCompletionRequestUserMessageContent::Text(t) => t,
        ChatCompletionRequestUserMessageContent::Array(parts) => parts
            .into_iter()
            .map(|p| match p {
                ChatCompletionRequestUserMessageContentPart::Text(t) => t.text,
                ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                    format!("[image_url:{}]", img.image_url.url)
                }
                ChatCompletionRequestUserMessageContentPart::InputAudio(_) => {
                    "[input_audio]".into()
                }
                ChatCompletionRequestUserMessageContentPart::File(_) => "[file]".into(),
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}
