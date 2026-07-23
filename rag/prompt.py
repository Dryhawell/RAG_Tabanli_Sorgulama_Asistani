from typing import List
from .types import RetrievedChunk

STRICT_RULES_TR = (
    "- Sadece verilen bağlamı kullan. Genel bilgi kullanma.\n"
    "- Yanıt bağlam tarafından tam desteklenmiyorsa: 'Bu bilgi dokümanda bulunmamaktadır' de.\n"
    "- Önemli ifadeleri alıntıla ve [dosya: {source_file}, sayfa: {page}, chunk: {chunk_id}] şeklinde atıf ver.\n"
    "- Kısa ve gerçekçi ol."
)

def render_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for c in chunks:
        meta = c.metadata
        heading = f" | Başlık: {meta.heading}" if meta.heading else ""
        header = (
            f"[Kaynak: {meta.source_file} | Sayfa: {meta.page_start}-{meta.page_end}"
            f"{heading} | Chunk: {meta.chunk_id} | Skor: {c.score:.3f}]"
        )
        parts.append(header)
        parts.append(c.text)
        parts.append("\n---\n")
    return "\n".join(parts)


def build_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    *,
    tools_context: str = "",
) -> str:
    ctx = render_context(chunks)
    tools_block = ""
    if tools_context and tools_context.strip():
        tools_block = f"\nAraç çıktıları (doküman dışı yardımcı bilgi):\n{tools_context.strip()}\n\n"
    prompt = (
        "Sen sadece doküman bağlamına dayalı cevap veren bir asistansın. Kurallara harfiyen uy.\n\n"
        f"Bağlam:\n{ctx}\n\n"
        f"{tools_block}"
        f"Soru:\n{question}\n\n"
        f"Kurallar:\n{STRICT_RULES_TR}\n\n"
        "Yanıt:\n"
    )
    return prompt
