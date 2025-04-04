import json
from utils.openai_client import client
from agents import weather_agent, email_agent, resume_agent

def process_ai_request(user_input: str):
    if resume_agent.is_resume_query(user_input):
        return resume_agent.query_rag(user_input)

    system_prompt = "You are an AI assistant that determines user intent and executes appropriate functions."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for provided coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to the given address.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_email": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to_email", "subject", "body"],
                },
            },
        }
    ]

    completion = client.chat.completions.create(model="gpt-4o", messages=messages, tools=tools)

    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            messages.append(completion.choices[0].message)

            if name == "get_weather":
                result = weather_agent.get_weather(**args)

            elif name == "send_email":
                result = email_agent.send_email(**args)

            else:
                result = {"error": "Unknown tool called."}

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

        completion_2 = client.chat.completions.create(model="gpt-4o", messages=messages)
        return {"response": completion_2.choices[0].message.content}

    else:
        completion_2 = client.chat.completions.create(model="gpt-4o", messages=messages)
        return {"response": completion_2.choices[0].message.content}
