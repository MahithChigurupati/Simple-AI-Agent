from utils.openai_client import client
from services import rag_service

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

def query_rag(user_input: str):
    return rag_service.query_rag(user_input)
