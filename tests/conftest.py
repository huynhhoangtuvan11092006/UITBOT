"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from app.ingestion.document import build_document, concatenate_pages
from app.ingestion.models import ExtractedPage
from app.ingestion.structure import parse_structure

SAMPLE_REGULATION = """Chương I
NHỮNG QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
1. Quy định này áp dụng cho sinh viên đại học chính quy của Trường Đại học Công nghệ Thông tin.
2. Các chương trình liên kết thực hiện theo thỏa thuận riêng đã được hiệu trưởng phê duyệt.

Điều 2. Đối tượng áp dụng
Văn bản này áp dụng đối với sinh viên, giảng viên và đơn vị đào tạo.

Điều 3. Giải thích từ ngữ
1. Tín chỉ là đơn vị dùng để đo lường khối lượng học tập.
2. Học kỳ chính có thời gian tối thiểu là 15 tuần.
"""


@pytest.fixture
def sample_pages() -> list[ExtractedPage]:
    return [
        ExtractedPage(
            source_file="quy_che_dao_tao.pdf",
            page=1,
            text=SAMPLE_REGULATION,
            extraction_method="text",
            ocr_confidence=None,
        )
    ]


@pytest.fixture
def sample_document(sample_pages: list[ExtractedPage]):
    text, _ = concatenate_pages(sample_pages)
    return build_document(sample_pages, parse_structure(text))


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
