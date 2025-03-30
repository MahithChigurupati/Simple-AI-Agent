from fastapi import APIRouter
from schemas.ai import UserRequest, AIResponse
from services.ai_service import process_ai_request

router = APIRouter()


@router.post("/ai-agent", response_model=AIResponse)
async def ai_agent(request: UserRequest):
    response = process_ai_request(request.input_text)
    return response
