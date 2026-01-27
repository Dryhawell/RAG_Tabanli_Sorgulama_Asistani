from pydantic import BaseModel
from typing import Optional

class ChunkMetadata(BaseModel):
    source_file: str
    chunk_id: int
    page_start: int
    page_end: int
    word_count: int
    heading: Optional[str] = None

class RetrievedChunk(BaseModel):
    chunk_id: int
    score: float
    text: str
    metadata: ChunkMetadata
