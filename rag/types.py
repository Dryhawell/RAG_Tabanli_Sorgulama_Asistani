from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    source_file: str
    chunk_id: int
    page_start: int
    page_end: int
    word_count: int
    heading: Optional[str] = None
    folder: str = ""
    tags: List[str] = Field(default_factory=list)
    chunk_uid: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    score: float
    text: str
    metadata: ChunkMetadata
