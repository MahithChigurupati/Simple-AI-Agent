from pydantic import BaseModel


class UserRequest(BaseModel):
    input_text: str


class AIResponse(BaseModel):
    response: str
