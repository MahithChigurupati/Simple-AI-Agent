from fastapi import APIRouter, HTTPException
from services.embedding_service import process_pdfs_and_create_embeddings

router = APIRouter()

@router.get("/create-embeddings/")
async def generate_embeddings():
    try:
        response = process_pdfs_and_create_embeddings()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
