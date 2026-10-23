from app.ingestion.ocr import NoneOcrEngine
from app.ingestion.quality import needs_ocr


def test_normal_vietnamese_page_does_not_need_ocr() -> None:
    text = (
        "Sinh viên phải hoàn thành chương trình đào tạo theo quy chế của trường "
        "và đảm bảo đủ số tín chỉ tối thiểu của ngành học."
    )
    assert needs_ocr(text) is False


def test_empty_page_needs_ocr() -> None:
    assert needs_ocr("   ") is True


def test_replacement_chars_need_ocr() -> None:
    text = "Nội dung bị lỗi font " + ("\ufffd" * 20) + " còn lại một ít chữ Việt."
    assert needs_ocr(text) is True


def test_garbled_letters_need_ocr() -> None:
    text = "Это страница с поврежденной кодировкой шрифта " * 4
    assert needs_ocr(text) is True


def test_none_ocr_engine_returns_empty() -> None:
    result = NoneOcrEngine().ocr_image(b"not-an-image")
    assert result.text == ""
    assert result.confidence is None
