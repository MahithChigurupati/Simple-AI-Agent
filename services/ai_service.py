import json
import requests
import os
from pydantic import BaseModel, Field
from utils.openai_client import client
from sendgrid import SendGridAPIClient
import sendgrid
from sendgrid.helpers.mail import Mail
from services.rag_service import query_rag  # Import RAG system


SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)

# Define expected response formats
class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in Celsius for the given location.")
    response: str = Field(description="A natural language response to the user's question.")

class EmailResponse(BaseModel):
    status: str = Field(description="The status of the email sending operation.")
    response: str = Field(description="A response indicating the result of the email operation.")
    
    
def is_resume_query(user_input: str):
    classification_prompt = (
        "You are a classifier that determines if a given question is about a person's resume, career, or work experience. "
        "Answer with 'yes' or 'no'.\n\n"
        f"Question: {user_input}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": classification_prompt}]
    )

    return response.choices[0].message.content.lower().strip() == "yes"


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

# Function to send an email using SendGrid
def send_email(to_email: str, subject: str, body: str):
    sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
    message = Mail(
        from_email="mahithchigurupati@gmail.com",
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        return {
            "status": "success" if response.status_code == 202 else "failure",
            "response": "Email sent successfully." if response.status_code == 202 else "Failed to send email.",
        }
    except Exception as e:
        return {"status": "error", "response": str(e)}

# AI processing function
def process_ai_request(user_input: str):
    # First, check if this is a resume-related question
    if is_resume_query(user_input):
        return query_rag(user_input)  # Call RAG system for resume questions

    # Otherwise, follow normal tool-based processing
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
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    # First OpenAI API call: Determine intent
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

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
                response_format = WeatherResponse

            elif name == "send_email":
                result = send_email(**args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
                response_format = EmailResponse

            else:
                response_format = None

        # Second OpenAI API call: Generate final response
        completion_2 = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            response_format=response_format if response_format else None,
        )

        return completion_2.choices[0].message.parsed

    else:
        # No tool needed, return AI response
        completion_2 = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        return {"response": completion_2.choices[0].message.content}