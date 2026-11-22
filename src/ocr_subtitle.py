"""Native Windows OCR fallback for image-based (PGS/VobSub) and hardcoded subtitles."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from PIL import Image, ImageFilter, ImageGrab, ImageOps
from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import (
    BitmapAlphaMode,
    BitmapPixelFormat,
    SoftwareBitmap,
)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

_STATUS_NOISE_RE = re.compile(
    r"^(?:paused|playing|stopped|muted|\d{1,2}:\d{2}(?::\d{2})?(?:\s*/\s*\d{1,2}:\d{2}(?::\d{2})?)?)$",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")


def _create_ocr_engine() -> Optional[OcrEngine]:
    """Create a Windows OcrEngine preferring English, falling back to user profile."""
    for tag in ("en-US", "en-GB"):
        try:
            lang = Language(tag)
            if OcrEngine.is_language_supported(lang):
                engine = OcrEngine.try_create_from_language(lang)
                if engine is not None:
                    return engine
        except Exception:
            continue
    return OcrEngine.try_create_from_user_profile_languages()


async def _run_ocr_async(bitmap: SoftwareBitmap) -> list[str]:
    engine = _create_ocr_engine()
    if engine is None:
        return []
    result = await engine.recognize_async(bitmap)
    lines: list[str] = []
    for line in result.lines:
        txt = _WHITESPACE_RE.sub(" ", (line.text or "")).strip()
        if txt and not _STATUS_NOISE_RE.match(txt):
            lines.append(txt)
    return lines


def _alpha_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _boost_subtitle_contrast(img: Image.Image) -> Image.Image:
    """Isolate bright subtitle text on a crisp white background for difficult frames."""
    gray = img.convert("L")
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    # Invert so bright subtitle text becomes dark text on pure white background
    binary = contrasted.point(lambda p: 0 if p >= 180 else 255)
    return binary.filter(ImageFilter.SHARPEN)


def _recognize_single(img: Image.Image) -> str:
    rgba = img.convert("RGBA")
    max_dim = int(getattr(OcrEngine, "max_image_dimension", 4096) or 4096)
    if rgba.width > max_dim or rgba.height > max_dim:
        scale = min(max_dim / rgba.width, max_dim / rgba.height)
        new_size = (max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale)))
        rgba = rgba.resize(new_size, Image.Resampling.BICUBIC)

    writer = DataWriter()
    writer.write_bytes(rgba.tobytes())
    buf = writer.detach_buffer()

    bitmap = SoftwareBitmap.create_copy_from_buffer(
        buf,
        BitmapPixelFormat.RGBA8,
        rgba.width,
        rgba.height,
        BitmapAlphaMode.PREMULTIPLIED,
    )
    lines = asyncio.run(_run_ocr_async(bitmap))
    return _WHITESPACE_RE.sub(" ", " ".join(lines)).strip()


def recognize_image_text(img: Image.Image) -> str:
    """Run Windows built-in OCR on a PIL Image with a contrast-boosted fallback pass."""
    if img.width <= 0 or img.height <= 0:
        return ""

    first = _recognize_single(img)
    if _alpha_count(first) >= 3:
        return first

    boosted = _boost_subtitle_contrast(img)
    second = _recognize_single(boosted)
    return second if _alpha_count(second) > _alpha_count(first) else first


def capture_subtitle_region(phys_rect: tuple[int, int, int, int]) -> Image.Image:
    """Capture the bottom subtitle band of the MPC-HC window in physical coordinates."""
    left, top, right, bottom = phys_rect
    w = max(1, right - left)
    h = max(1, bottom - top)

    crop_left = left + int(w * 0.05)
    crop_right = right - int(w * 0.05)
    crop_top = top + int(h * 0.72)
    crop_bottom = bottom - int(h * 0.02)

    if crop_right <= crop_left:
        crop_left, crop_right = left, right
    if crop_bottom <= crop_top:
        crop_top, crop_bottom = top, bottom

    img = ImageGrab.grab(bbox=(crop_left, crop_top, crop_right, crop_bottom), all_screens=True)
    if img.height < 180 and img.height > 0:
        img = img.resize((img.width * 2, img.height * 2), Image.Resampling.BICUBIC)
    return img


def ocr_mpc_subtitle(phys_rect: tuple[int, int, int, int]) -> str:
    """Capture the bottom subtitle strip of MPC-HC and extract text via Windows OCR."""
    img = capture_subtitle_region(phys_rect)
    return recognize_image_text(img)
