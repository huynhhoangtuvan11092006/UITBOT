"""OCR engines. Tesseract is the default; other engines can implement OcrEngine."""

from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass
from typing import Protocol

from PIL import Image, ImageOps

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | None


class OcrEngine(Protocol):
    name: str

    def ocr_image(self, image_bytes: bytes) -> OcrResult: ...


class NoneOcrEngine:
    name = "none"

    def ocr_image(self, image_bytes: bytes) -> OcrResult:
        del image_bytes
        return OcrResult(text="", confidence=None)


class TesseractOcrEngine:
    name = "tesseract"

    def ocr_image(self, image_bytes: bytes) -> OcrResult:
        import pytesseract

        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.grayscale(image)
        data = pytesseract.image_to_data(
            image,
            lang="vie",
            output_type=pytesseract.Output.DICT,
        )
        texts = [word for word in data.get("text", []) if word and word.strip()]
        confs = []
        for value in data.get("conf", []):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                confs.append(number)
        text = " ".join(texts)
        confidence = sum(confs) / len(confs) if confs else None
        return OcrResult(text=text, confidence=confidence)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def get_ocr_engine(name: str | None = None) -> OcrEngine:
    engine_name = (name or settings.ocr_engine).lower()
    if engine_name == "none":
        return NoneOcrEngine()
    if engine_name == "tesseract":
        if not tesseract_available():
            logger.warning(
                "OCR_ENGINE=tesseract but the tesseract binary was not found; using none"
            )
            return NoneOcrEngine()
        return TesseractOcrEngine()
    raise ValueError(f"Unsupported OCR engine: {engine_name}")
