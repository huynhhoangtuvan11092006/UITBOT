CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    chunk_set TEXT NOT NULL,
    source_file TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    chapter TEXT,
    article TEXT,
    article_title TEXT,
    text TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    ocr_confidence DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_chunk_set_idx ON chunks (chunk_set);
CREATE INDEX IF NOT EXISTS chunks_source_file_idx ON chunks (source_file);
CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks (article);

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT,
    rating INTEGER,
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
