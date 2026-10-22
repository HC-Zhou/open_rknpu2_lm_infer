use crate::api::{self, ToolCallParser};
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn request(parser: ToolCallParser, stream: bool, choice: &str, output: &str) -> String {
    let preserve = parser == ToolCallParser::Minicpm5 && !stream && choice != "none";
    let body = json!({"model":"mock", "stream":stream, "tool_choice":choice,
        "messages":[{"role":"user","content":format!("{}\n{output}", if preserve { "preserve" } else { "default" })}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}]});
    let response = api::router_with_parser(crate::engine::Engine, None, parser)
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap()
}

#[tokio::test]
async fn xml_response_does_not_validate_tool_name_or_schema() {
    let output = "before<function name='unknown'><param name='extra'>[1,true]</param></function><function name='f'><param name='x'>not an integer</param></function>after";
    let response: Value =
        serde_json::from_str(&request(ToolCallParser::Minicpm5, false, "auto", output).await)
            .unwrap();
    let choice = &response["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let message = &choice["message"];
    assert_eq!(message["content"], "beforeafter");
    let calls = message["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "unknown");
    assert_eq!(
        serde_json::from_str::<Value>(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"extra":[1,true]})
    );
    assert_eq!(
        serde_json::from_str::<Value>(calls[1]["function"]["arguments"].as_str().unwrap()).unwrap(),
        json!({"x":"not an integer"})
    );
    assert_ne!(calls[0]["id"], calls[1]["id"]);
}

#[tokio::test]
async fn selection_none_and_streaming_keep_existing_behavior() {
    let xml = "<function name='f'><param name='x'>1</param></function>";
    let json_call = "<tool_call>{\"name\":\"f\",\"arguments\":{\"x\":1}}</tool_call>";
    for protocol in [ToolCallParser::ToolCallJson, ToolCallParser::Minicpm5] {
        for output in [xml, json_call] {
            for choice in ["auto", "none"] {
                let response: Value =
                    serde_json::from_str(&request(protocol, false, choice, output).await).unwrap();
                let expected =
                    choice == "auto" && ((protocol == ToolCallParser::Minicpm5) == (output == xml));
                assert_eq!(
                    response["choices"][0]["message"]
                        .get("tool_calls")
                        .is_some(),
                    expected
                );
                if !expected {
                    assert_eq!(response["choices"][0]["message"]["content"], output);
                }
                let sse = request(protocol, true, choice, output).await;
                assert_eq!(
                    sse.contains("\"tool_calls\":["),
                    choice == "auto" && output == json_call
                );
                assert!(sse.contains("data: [DONE]"));
            }
        }
    }
}

#[tokio::test]
async fn thinking_and_bad_xml_are_preserved() {
    let output = "<think>reason</think><function name='f'><param name='x'>[oops]</param></function><function name='f'>";
    let response: Value =
        serde_json::from_str(&request(ToolCallParser::Minicpm5, false, "auto", output).await)
            .unwrap();
    let msg = &response["choices"][0]["message"];
    assert_eq!(msg["reasoning_content"], "reason");
    assert_eq!(msg["content"], "<function name='f'>");
    assert_eq!(
        msg["tool_calls"][0]["function"]["arguments"],
        "{\"x\":\"[oops]\"}"
    );
}

#[tokio::test]
async fn default_router_and_requests_without_tools_do_not_preserve_xml_tokens() {
    let xml = "<function name='f'><param name='x'>1</param></function>";
    for default_router in [false, true] {
        for chat in [false, true] {
            let router = if default_router {
                api::router(crate::engine::Engine, None)
            } else {
                api::router_with_parser(crate::engine::Engine, None, ToolCallParser::Minicpm5)
            };
            let prompt = format!("default\n{xml}");
            let body = if chat {
                json!({"model":"mock", "messages":[{"role":"user","content":prompt}]})
            } else {
                json!({"model":"mock", "prompt":prompt})
            };
            let response = router
                .oneshot(
                    Request::post(if chat {
                        "/v1/chat/completions"
                    } else {
                        "/v1/completions"
                    })
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let result: Value =
                serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                    .unwrap();
            if chat {
                assert_eq!(result["choices"][0]["message"]["content"], xml);
                assert!(result["choices"][0]["message"].get("tool_calls").is_none());
            } else {
                assert_eq!(result["choices"][0]["text"], xml);
            }
        }
    }
}
