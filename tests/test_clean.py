from app.ingestion.clean import clean_text, detect_repeated_headers_footers, strip_repeated_lines


def test_clean_text_nfc_and_nbsp() -> None:
    raw = "Quy\u00a0chế\nđào tạo"
    cleaned = clean_text(raw)
    assert "\u00a0" not in cleaned
    assert "Quy chế" in cleaned


def test_clean_text_joins_hyphenated_line_breaks() -> None:
    cleaned = clean_text("sinh-\nviên")
    assert cleaned == "sinhviên"


def test_clean_text_keeps_dieu_on_its_own_line() -> None:
    cleaned = clean_text("Điều 1. Phạm vi\n1. Nội dung")
    assert "Điều 1. Phạm vi" in cleaned
    assert cleaned.splitlines()[0].startswith("Điều 1")


def test_repeated_header_stripped() -> None:
    pages = [
        "TRƯỜNG UIT\nNội dung A\nTrang 1",
        "TRƯỜNG UIT\nNội dung B\nTrang 2",
        "TRƯỜNG UIT\nNội dung C\nTrang 3",
    ]
    repeated = detect_repeated_headers_footers(pages)
    assert "TRƯỜNG UIT" in repeated
    stripped = strip_repeated_lines(pages[0], repeated)
    assert "TRƯỜNG UIT" not in stripped
    assert "Nội dung A" in stripped
