"""Çoklu doküman özetleme ve karşılaştırma."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from rag.types import ChunkMetadata, RetrievedChunk

GenerateFn = Callable[[str], str]

SUMMARIZE_PROMPT = """Aşağıdaki doküman parçalarına dayanarak kısa bir özet yaz.
Yalnızca verilen bağlamı kullan. Uydurma.

Kaynak: {source}
Bağlam:
{context}

Özet:"""

COMPARE_PROMPT = """Aşağıdaki dokümanlardan alınan parçaları karşılaştır.
Soru / odak: {focus}

Kurallar:
- Yalnızca verilen bağlamı kullan.
- Benzerlikleri ve farkları maddeler halinde yaz.
- Her iddiada kaynak dosya adını belirt.
- Bağlamda yoksa bunu açıkça söyle.

Bağlam:
{context}

Karşılaştırma:"""


@dataclass
class SourceBundle:
    source_file: str
    chunks: List[RetrievedChunk]

    @property
    def text(self) -> str:
        parts = []
        for c in self.chunks:
            meta = c.metadata
            parts.append(
                f"[chunk {meta.chunk_id} p.{meta.page_start}-{meta.page_end}]\n{c.text}"
            )
        return "\n\n".join(parts)


def collect_source_bundles(
    index,
    sources: Sequence[str],
    *,
    max_chunks_per: int = 6,
) -> List[SourceBundle]:
    """İndeksten seçili kaynaklara ait chunk'ları toplar (sıra korunur)."""
    wanted = [s for s in sources if s]
    bundles: List[SourceBundle] = []
    for src in wanted:
        ids = index.ids_for_source(src)[:max_chunks_per]
        chunks: List[RetrievedChunk] = []
        for cid in ids:
            text = index._id_to_text.get(cid)
            meta = index._id_to_meta.get(cid)
            if text is None or meta is None:
                continue
            chunks.append(
                RetrievedChunk(chunk_id=cid, score=1.0, text=text, metadata=meta)
            )
        if chunks:
            bundles.append(SourceBundle(source_file=src, chunks=chunks))
    return bundles


def summarize_source(
    bundle: SourceBundle,
    generate_fn: GenerateFn,
) -> str:
    prompt = SUMMARIZE_PROMPT.format(source=bundle.source_file, context=bundle.text)
    return (generate_fn(prompt) or "").strip()


def summarize_sources(
    index,
    sources: Sequence[str],
    generate_fn: GenerateFn,
    *,
    max_chunks_per: int = 6,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for bundle in collect_source_bundles(index, sources, max_chunks_per=max_chunks_per):
        out[bundle.source_file] = summarize_source(bundle, generate_fn)
    return out


def compare_sources(
    index,
    sources: Sequence[str],
    generate_fn: GenerateFn,
    *,
    focus: str = "Ana noktaları ve farkları karşılaştır",
    max_chunks_per: int = 5,
) -> str:
    bundles = collect_source_bundles(index, sources, max_chunks_per=max_chunks_per)
    if len(bundles) < 2:
        raise ValueError("Karşılaştırma için en az 2 kaynak seçin")
    parts = []
    for b in bundles:
        parts.append(f"### {b.source_file}\n{b.text}")
    context = "\n\n".join(parts)
    prompt = COMPARE_PROMPT.format(focus=focus.strip() or "genel karşılaştırma", context=context)
    return (generate_fn(prompt) or "").strip()
