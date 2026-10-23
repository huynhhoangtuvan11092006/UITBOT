from app.ingestion.structure import find_clause_spans, parse_structure
from tests.conftest import SAMPLE_REGULATION


def test_parse_chapters_and_articles() -> None:
    regions = parse_structure(SAMPLE_REGULATION)
    articles = [region.article for region in regions if region.article]
    assert "Điều 1" in articles
    assert "Điều 2" in articles
    assert "Điều 3" in articles
    dieu1 = next(region for region in regions if region.article == "Điều 1")
    assert dieu1.chapter == "Chương I"
    assert dieu1.article_title == "Phạm vi điều chỉnh"
    assert "tín chỉ" not in SAMPLE_REGULATION[dieu1.start : dieu1.end]


def test_unstructured_text_has_single_region() -> None:
    regions = parse_structure("Không có điều khoản nào ở đây.")
    assert len(regions) == 1
    assert regions[0].article is None


def test_clause_spans() -> None:
    text = "Điều 1. Tiêu đề\n1. Khoản một.\n2. Khoản hai."
    regions = parse_structure(text)
    dieu1 = next(region for region in regions if region.article == "Điều 1")
    spans = find_clause_spans(text, dieu1.start, dieu1.end)
    assert len(spans) >= 2
