from app.ingestion.chunkers.fixed import chunk_fixed
from app.ingestion.chunkers.sentence import chunk_sentence
from app.ingestion.chunkers.split import recursive_split
from app.ingestion.chunkers.structured import chunk_structured
from app.ingestion.document import build_document, concatenate_pages
from app.ingestion.models import ExtractedPage
from app.ingestion.structure import parse_structure


def test_recursive_split_respects_size() -> None:
    text = "alpha " * 80
    spans = recursive_split(text, chunk_size=40, chunk_overlap=10)
    assert spans
    for start, end in spans:
        assert end > start
        assert end - start <= 80


def test_fixed_chunker_attaches_article(sample_document) -> None:
    chunks = chunk_fixed(sample_document, chunk_set="fixed_80_10", chunk_size=80, chunk_overlap=10)
    assert chunks
    assert all(chunk.chunk_set == "fixed_80_10" for chunk in chunks)
    assert any(chunk.article == "Điều 1" for chunk in chunks)
    assert chunks[0].source_file == "quy_che_dao_tao.pdf"
    assert chunks[0].page_start >= 1


def test_sentence_chunker(sample_document) -> None:
    chunks = chunk_sentence(
        sample_document,
        chunk_set="sentence_200_1",
        chunk_size=200,
        sentence_overlap=1,
    )
    assert chunks
    joined = " ".join(chunk.text for chunk in chunks)
    assert "Phạm vi điều chỉnh" in joined


def test_structured_chunker_keeps_dieu(sample_document) -> None:
    chunks = chunk_structured(sample_document, chunk_set="structured_800")
    articles = {chunk.article for chunk in chunks}
    assert "Điều 1" in articles
    assert "Điều 2" in articles
    dieu1 = next(chunk for chunk in chunks if chunk.article == "Điều 1")
    assert "Phạm vi điều chỉnh" in dieu1.text or "chính quy" in dieu1.text


def test_structured_falls_back_when_no_articles() -> None:
    pages = [
        ExtractedPage(
            source_file="plain.pdf",
            page=1,
            text="Đây là đoạn văn không có cấu trúc điều khoản. " * 20,
            extraction_method="text",
            ocr_confidence=None,
        )
    ]
    text, _ = concatenate_pages(pages)
    document = build_document(pages, parse_structure(text))
    chunks = chunk_structured(document, chunk_set="structured_80", chunk_size=80)
    assert chunks
    assert all(chunk.article is None for chunk in chunks)
