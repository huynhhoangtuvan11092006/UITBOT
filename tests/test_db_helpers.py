from __future__ import annotations

from app.db.embeddings import embedding_table_name, model_slug
from app.ingestion.embed_and_load import upsert_chunks
from app.ingestion.models import Chunk


def test_model_slug_and_table_name() -> None:
    assert model_slug("BAAI/bge-m3") == "baai_bge_m3"
    assert embedding_table_name("BAAI/bge-m3") == "chunk_embeddings_baai_bge_m3"


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple | None]] = []

    def execute(self, query, params=None) -> None:  # noqa: ANN001
        self.statements.append((str(query), params))

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()

    def cursor(self) -> _FakeCursor:
        return self.cursor_obj


def test_upsert_sql_is_idempotent() -> None:
    chunk = Chunk(
        chunk_id="file.pdf::fixed_800_150::0000",
        chunk_set="fixed_800_150",
        source_file="file.pdf",
        page_start=1,
        page_end=1,
        chapter=None,
        article="Điều 1",
        article_title="Test",
        text="Nội dung",
        extraction_method="text",
        ocr_confidence=None,
    )
    conn = _FakeConn()
    upsert_chunks(conn, [chunk], [[0.1, 0.2]], embedding_table="chunk_embeddings_baai_bge_m3")
    sql_text = " ".join(query for query, _ in conn.cursor_obj.statements)
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql_text
    assert "DELETE FROM chunks" in sql_text
    assert sql_text.lower().count("on conflict") >= 2
