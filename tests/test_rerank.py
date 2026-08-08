from rag.rerank import LexicalReranker, rerank_chunks
from rag.types import ChunkMetadata, RetrievedChunk


def _chunk(cid: int, text: str, source: str = "a.txt") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        score=0.1,
        text=text,
        metadata=ChunkMetadata(
            source_file=source,
            chunk_id=cid,
            page_start=1,
            page_end=1,
            word_count=len(text.split()),
        ),
    )


def test_lexical_rerank_prefers_relevant():
    chunks = [
        _chunk(0, "arabalar otoyolda hızlanır"),
        _chunk(1, "kedi süt içer ve fare yakalar"),
        _chunk(2, "hava durumu raporu"),
    ]
    ranked = rerank_chunks(
        "kedi süt",
        chunks,
        top_k=2,
        reranker=LexicalReranker(),
    )
    assert ranked[0].chunk_id == 1
    assert ranked[0].score >= ranked[1].score
