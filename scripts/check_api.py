#!/usr/bin/env python3
"""Exercise a running inference service using the official OpenAI Python SDK."""
import argparse
import json
import time
from pathlib import Path
from openai import OpenAI, APIStatusError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--api-key", default="board-test-key")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=120, max_retries=0)
    model = client.models.list().data[0].id
    report = {"model":model,"checks":{}}
    messages = [{"role":"user","content":"只回复一个汉字：好"}]
    common = {"model":model,"temperature":0,"max_tokens":24,"extra_body":{"enable_thinking":False}}
    result = client.chat.completions.create(messages=messages,**common)
    assert result.choices and result.choices[0].message.content
    assert result.usage.prompt_tokens>0 and result.usage.completion_tokens>0
    report["checks"]["chat"] = result.model_dump()

    stream = client.chat.completions.create(messages=messages,stream=True,stream_options={"include_usage":True},**common)
    pieces,usage,finish=[],None,None
    with stream:
        for chunk in stream:
            if chunk.choices:
                pieces.append(chunk.choices[0].delta.content or "")
                finish = chunk.choices[0].finish_reason or finish
            if chunk.usage: usage=chunk.usage.model_dump()
    assert "".join(pieces)==result.choices[0].message.content
    assert finish and usage
    report["checks"]["stream"]={"text":"".join(pieces),"usage":usage,"finish_reason":finish}

    messages += [{"role":"assistant","content":result.choices[0].message.content},{"role":"user","content":"再回复一个字：是"}]
    second = client.chat.completions.create(messages=messages,**common)
    assert second.usage.prompt_tokens_details.cached_tokens>0
    report["checks"]["prefix_cache"] = second.model_dump()

    completion = client.completions.create(model=model,prompt="1, 2, 3,",temperature=0,max_tokens=8)
    assert completion.choices and completion.usage.completion_tokens>0
    report["checks"]["completion"] = completion.model_dump()

    tools=[{"type":"function","function":{"name":"get_weather","description":"查询城市的天气。用户查询天气时必须调用此工具。","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]
    tool_messages=[{"role":"user","content":"请调用get_weather查询杭州的天气，不要自己编造天气信息。"}]
    tool_result=client.chat.completions.create(model=model,messages=tool_messages,tools=tools,tool_choice="auto",temperature=0,max_tokens=128,extra_body={"enable_thinking":False})
    calls=tool_result.choices[0].message.tool_calls
    assert calls and tool_result.choices[0].finish_reason=="tool_calls"
    assert calls[0].function.name=="get_weather"
    assert json.loads(calls[0].function.arguments)["city"]
    report["checks"]["tools"] = tool_result.model_dump()
    tool_stream=client.chat.completions.create(model=model,messages=tool_messages,tools=tools,tool_choice="auto",stream=True,temperature=0,max_tokens=128,extra_body={"enable_thinking":False})
    streamed_calls,tool_finish=[],None
    with tool_stream:
        for chunk in tool_stream:
            if chunk.choices:
                choice=chunk.choices[0]
                if choice.delta.tool_calls: streamed_calls.extend(call.model_dump() for call in choice.delta.tool_calls)
                tool_finish=choice.finish_reason or tool_finish
    assert streamed_calls and tool_finish=="tool_calls"
    assert streamed_calls[0]["function"]["name"]=="get_weather"
    assert json.loads(streamed_calls[0]["function"]["arguments"])["city"]
    report["checks"]["stream_tools"]={"calls":streamed_calls,"finish_reason":tool_finish}
    tool_messages += [tool_result.choices[0].message.model_dump(exclude_none=True),{"role":"tool","tool_call_id":calls[0].id,"content":"杭州：晴，25摄氏度。"}]
    tool_answer=client.chat.completions.create(model=model,messages=tool_messages,tools=tools,tool_choice="none",temperature=0,max_tokens=48,extra_body={"enable_thinking":False})
    assert tool_answer.choices[0].message.content
    report["checks"]["tool_history"] = tool_answer.model_dump()

    thinking=client.chat.completions.create(model=model,messages=[{"role":"user","content":"计算1+1，只需给出数字答案。"}],temperature=0,max_tokens=256,extra_body={"enable_thinking":True})
    thought=thinking.choices[0].message.model_dump()
    assert thought.get("reasoning_content")
    assert thinking.choices[0].finish_reason in {"stop","length"}
    if thinking.choices[0].finish_reason=="stop": assert thought.get("content")
    report["checks"]["reasoning"] = thinking.model_dump()

    for label,extra in [("json_mode",{"response_format":{"type":"json_object"}}),("context_overflow",{"max_tokens":100000})]:
        request={**common,**extra}
        try:
            client.chat.completions.create(messages=[{"role":"user","content":"你好"}],**request)
            raise AssertionError(label+" should reject")
        except APIStatusError as error:
            assert error.status_code==400
            report["checks"][label]={"status":error.status_code,"message":error.message}

    cancelled=client.chat.completions.create(model=model,messages=[{"role":"user","content":"写一篇很长的游记。"}],stream=True,max_tokens=512,temperature=0,extra_body={"enable_thinking":False})
    with cancelled:
        for chunk in cancelled:
            if chunk.choices and chunk.choices[0].delta.content: break
    begin=time.monotonic()
    recovery=client.chat.completions.create(messages=[{"role":"user","content":"只回复：好"}],**common)
    assert recovery.choices[0].message.content
    report["checks"]["disconnect_recovery"]={"seconds":time.monotonic()-begin,"result":recovery.model_dump()}

    streams=[]
    try:
        for _ in range(8):
            try:
                streams.append(client.chat.completions.create(model=model,messages=[{"role":"user","content":"写一篇非常长的游记。"}],stream=True,max_tokens=512,temperature=0,extra_body={"enable_thinking":False}))
            except APIStatusError as error:
                assert error.status_code==503
                report["checks"]["queue_backpressure"]={"accepted":len(streams),"status":error.status_code}
                break
        assert "queue_backpressure" in report["checks"]
    finally:
        for stream in streams: stream.close()
    deadline=time.monotonic()+5
    while True:
        try:
            final=client.chat.completions.create(messages=[{"role":"user","content":"只回复：好"}],**common)
            break
        except APIStatusError as error:
            if error.status_code!=503 or time.monotonic()>=deadline: raise
            time.sleep(0.1)
    assert final.choices[0].message.content
    report["checks"]["queue_recovery"]=final.model_dump()

    parameters=client.chat.completions.create(
        model=model,
        messages=[{"role":"developer","content":"只回复一个字：好"},{"role":"user","content":[{"type":"text","text":"开始"}]}],
        seed=-1,top_p=0,temperature=0.7,max_completion_tokens=8,
        extra_body={"enable_thinking":False},
    )
    assert parameters.choices[0].message.content and parameters.usage.completion_tokens>0
    report["checks"]["request_parameters"]=parameters.model_dump()

    report["passed"]=True
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"passed":True,"model":model,"checks":list(report["checks"])},ensure_ascii=False))


if __name__=="__main__":
    main()
