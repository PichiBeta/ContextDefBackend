from fastapi import APIRouter
from app.services.chunking import splitter 
router = APIRouter()


@router.post("/chunk_text")
async def chunk_text(body: str):
    chunks = splitter.split_text(body)
    return {"chunks": chunks, "number_of_chunks": len(chunks)}
    

    