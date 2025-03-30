import json
import requests
from pydantic import BaseModel, Field
from utils.openai_client import client


# Define the expected response format
class WeatherResponse(BaseModel):
    temperature: float = Field(
        description="The current temperature in Celsius for the given location."
    )
    response: str = Field(
        description="A natural language response to the user's question."
    )


# Function to get weather data
def get_weather(latitude: float, longitude: float):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    weather_data = response.json().get("current", {})
    return {
        "temperature": weather_data.get("temperature_2m", "N/A"),
        "response": f"The current temperature is {weather_data.get('temperature_2m', 'N/A')}°C with wind speed {weather_data.get('wind_speed_10m', 'N/A')} m/s.",
    }


# AI processing function
def process_ai_request(user_input: str):
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
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    # First OpenAI API call: Determine if a tool needs to be called
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    # Process tool calls if detected
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls:
        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            messages.append(completion.choices[0].message)

            if name == "get_weather":
                result = get_weather(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

    # Second OpenAI API call: Parse structured response
    completion_2 = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        response_format=WeatherResponse,
    )

    # Return parsed response
    return completion_2.choices[0].message.parsed
