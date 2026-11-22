"""Subtitle discovery, background ffmpeg extraction, caching, and timestamp lookup."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pysubs2

TEXT_SUBTITLE_CODECS = {
    "subrip",
    "srt",
    "ass",
    "ssa",
    "mov_text",
    "webvtt",
    "text",
    "utf8",
}

IMAGE_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle",
    "dvd_subtitle",
    "dvb_subtitle",
    "xsub",
}

EXTERNAL_SUB_EXTENSIONS = (".srt", ".ass", ".ssa", ".vtt")

_ASS_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

_CREATE_NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def clean_subtitle_text(raw: str) -> str:
    """Strip ASS/HTML formatting tags and normalize line breaks/whitespace."""
    if not raw:
        return ""
    text = raw.replace(r"\N", " ").replace(r"\n", " ").replace(r"\h", " ")
    text = _ASS_TAG_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


@dataclass
class SubtitleCue:
    start_ms: int
    end_ms: int
    text: str


@dataclass
class SubtitleLoadResult:
    filepath: str
    status: str  # "ready" | "no_subtitles" | "image_only" | "error"
    source_kind: str = ""  # "external" | "embedded" | ""
    message: str = ""
    cues: list[SubtitleCue] = field(default_factory=list)


class SubtitleExtractor:
    """Extracts and caches text subtitles from external files or embedded mkv/mp4 tracks."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: dict[str, SubtitleLoadResult] = {}

    @staticmethod
    def _normalize_key(filepath: str) -> str:
        return os.path.normcase(os.path.abspath(filepath))

    def get_cached_result(self, filepath: str) -> Optional[SubtitleLoadResult]:
        key = self._normalize_key(filepath)
        with self._lock:
            return self._cache.get(key)

    def clear_file(self, filepath: str) -> None:
        key = self._normalize_key(filepath)
        with self._lock:
            self._cache.pop(key, None)

    @staticmethod
    def find_external_subtitle(video_path: Path) -> Optional[Path]:
        """Look for matching external .srt/.ass/.ssa/.vtt files next to the video."""
        if not video_path.parent.exists():
            return None
        stem = video_path.stem
        # Exact stem matches first
        for ext in EXTERNAL_SUB_EXTENSIONS:
            candidate = video_path.with_suffix(ext)
            if candidate.is_file():
                return candidate
        # Language suffix matches like video.en.srt, video.eng.ass
        for lang in ("en", "eng", "english"):
            for ext in EXTERNAL_SUB_EXTENSIONS:
                candidate = video_path.parent / f"{stem}.{lang}{ext}"
                if candidate.is_file():
                    return candidate
        return None

    @staticmethod
    def _load_cues_from_file(sub_path: Path) -> list[SubtitleCue]:
        subs = pysubs2.load(str(sub_path))
        cues: list[SubtitleCue] = []
        for event in subs:
            if getattr(event, "is_comment", False):
                continue
            cleaned = clean_subtitle_text(event.text)
            if not cleaned:
                continue
            cues.append(
                SubtitleCue(
                    start_ms=int(event.start),
                    end_ms=int(event.end),
                    text=cleaned,
                )
            )
        cues.sort(key=lambda c: (c.start_ms, c.end_ms))
        return cues

    @staticmethod
    def _probe_subtitle_streams(video_path: Path) -> list[dict]:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "s",
            str(video_path),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_CREATE_NO_WINDOW,
            timeout=20,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []
        try:
            payload = json.loads(proc.stdout)
            return payload.get("streams", []) or []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _pick_best_text_stream(streams: list[dict]) -> tuple[Optional[dict], bool]:
        """Return (best_text_stream, had_image_only_streams)."""
        text_streams: list[dict] = []
        has_image = False

        for s in streams:
            codec = (s.get("codec_name") or "").lower()
            if codec in TEXT_SUBTITLE_CODECS:
                text_streams.append(s)
            elif codec in IMAGE_SUBTITLE_CODECS:
                has_image = True

        if not text_streams:
            return None, has_image

        def _score(s: dict) -> tuple[int, int, int, int]:
            tags = {k.lower(): str(v).lower() for k, v in (s.get("tags") or {}).items()}
            lang = tags.get("language", "")
            title = tags.get("title", "")
            disp = s.get("disposition") or {}

            is_eng = 1 if lang in {"eng", "en", "english"} or "english" in title else 0
            is_default = 1 if disp.get("default") else 0
            not_forced = 0 if disp.get("forced") else 1
            not_commentary = 0 if "commentary" in title else 1
            return (is_eng, is_default, not_forced, not_commentary)

        best = max(text_streams, key=_score)
        return best, False

    def _disk_cache_path(self, video_path: Path) -> Path:
        stat = video_path.stat()
        raw_key = f"{os.path.normcase(str(video_path.resolve()))}|{stat.st_mtime_ns}"
        digest = hashlib.md5(raw_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.srt"

    def load_for_video(self, filepath: str) -> SubtitleLoadResult:
        """Load external or embedded subtitles for filepath, caching in memory and on disk."""
        if not filepath:
            return SubtitleLoadResult(filepath="", status="no_subtitles", message="No video loaded.")

        key = self._normalize_key(filepath)
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                return existing

        video_path = Path(filepath)
        if not video_path.is_file():
            res = SubtitleLoadResult(
                filepath=filepath,
                status="error",
                message=f"Video file not found: {filepath}",
            )
            with self._lock:
                self._cache[key] = res
            return res

        # 1. Check external subtitle file first
        ext_sub = self.find_external_subtitle(video_path)
        if ext_sub is not None:
            try:
                cues = self._load_cues_from_file(ext_sub)
                res = SubtitleLoadResult(
                    filepath=filepath,
                    status="ready" if cues else "no_subtitles",
                    source_kind="external",
                    message=f"Loaded external subtitle: {ext_sub.name}"
                    if cues
                    else f"External subtitle {ext_sub.name} had no dialogue lines.",
                    cues=cues,
                )
            except Exception as exc:
                res = SubtitleLoadResult(
                    filepath=filepath,
                    status="error",
                    message=f"Failed to parse external subtitle: {exc}",
                )
            with self._lock:
                self._cache[key] = res
            return res

        # 2. Check disk cache for previously extracted embedded subtitle
        cache_srt = self._disk_cache_path(video_path)
        if cache_srt.is_file() and cache_srt.stat().st_size > 0:
            try:
                cues = self._load_cues_from_file(cache_srt)
                if cues:
                    res = SubtitleLoadResult(
                        filepath=filepath,
                        status="ready",
                        source_kind="embedded",
                        message="Loaded cached embedded subtitles.",
                        cues=cues,
                    )
                    with self._lock:
                        self._cache[key] = res
                    return res
            except Exception:
                pass  # Fall through to re-extract

        # 3. Probe embedded streams via ffprobe
        try:
            streams = self._probe_subtitle_streams(video_path)
        except FileNotFoundError:
            res = SubtitleLoadResult(
                filepath=filepath,
                status="error",
                message="ffprobe/ffmpeg not found on PATH.",
            )
            with self._lock:
                self._cache[key] = res
            return res
        except Exception as exc:
            res = SubtitleLoadResult(
                filepath=filepath,
                status="error",
                message=f"Failed to probe subtitles: {exc}",
            )
            with self._lock:
                self._cache[key] = res
            return res

        if not streams:
            res = SubtitleLoadResult(
                filepath=filepath,
                status="no_subtitles",
                message="No embedded or external subtitle tracks found in this video.",
            )
            with self._lock:
                self._cache[key] = res
            return res

        best_stream, image_only = self._pick_best_text_stream(streams)
        if best_stream is None:
            msg = (
                "Embedded subtitles are image-based (PGS/VobSub) and cannot be extracted as text."
                if image_only
                else "No compatible text subtitle streams found."
            )
            res = SubtitleLoadResult(
                filepath=filepath,
                status="image_only" if image_only else "no_subtitles",
                message=msg,
            )
            with self._lock:
                self._cache[key] = res
            return res

        stream_index = int(best_stream["index"])
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            f"0:{stream_index}",
            "-c:s",
            "srt",
            str(cache_srt),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
                timeout=90,
            )
            if proc.returncode != 0 or not cache_srt.is_file() or cache_srt.stat().st_size == 0:
                err_detail = (proc.stderr or "").strip() or "ffmpeg returned empty output"
                res = SubtitleLoadResult(
                    filepath=filepath,
                    status="error",
                    message=f"Failed to extract embedded subtitle: {err_detail}",
                )
            else:
                cues = self._load_cues_from_file(cache_srt)
                res = SubtitleLoadResult(
                    filepath=filepath,
                    status="ready" if cues else "no_subtitles",
                    source_kind="embedded",
                    message=f"Extracted embedded track #{stream_index} ({len(cues)} lines)."
                    if cues
                    else "Embedded subtitle track contained no dialogue lines.",
                    cues=cues,
                )
        except Exception as exc:
            res = SubtitleLoadResult(
                filepath=filepath,
                status="error",
                message=f"Subtitle extraction error: {exc}",
            )

        with self._lock:
            self._cache[key] = res
        return res

    def get_subtitle_at(
        self, filepath: str, position_ms: int, buffer_ms: int = 500
    ) -> tuple[str, str]:
        """Return (active_text, context_sentence) at position_ms, or ('', '') if none."""
        result = self.get_cached_result(filepath)
        if result is None:
            result = self.load_for_video(filepath)
        if result.status != "ready" or not result.cues:
            return "", ""

        cues = result.cues
        # Prefer exact active cues first
        active_indices = [
            i for i, cue in enumerate(cues) if cue.start_ms <= position_ms <= cue.end_ms
        ]

        # If user paused right after a line vanished (within buffer_ms), pick the most recent cue
        if not active_indices and buffer_ms > 0:
            near = [
                (abs(position_ms - cue.end_ms), i)
                for i, cue in enumerate(cues)
                if cue.end_ms < position_ms and (position_ms - cue.end_ms) <= buffer_ms
            ]
            if not near:
                near = [
                    (abs(cue.start_ms - position_ms), i)
                    for i, cue in enumerate(cues)
                    if cue.start_ms > position_ms and (cue.start_ms - position_ms) <= min(200, buffer_ms)
                ]
            if near:
                near.sort(key=lambda item: item[0])
                active_indices = [near[0][1]]

        if not active_indices:
            return "", ""

        # Deduplicate identical simultaneous lines
        active_lines: list[str] = []
        for idx in active_indices:
            txt = cues[idx].text
            if txt not in active_lines:
                active_lines.append(txt)
        active_text = " ".join(active_lines).strip()

        # Build sentence context by optionally stitching an adjacent incomplete fragment
        first_idx = active_indices[0]
        last_idx = active_indices[-1]
        context_parts = list(active_lines)

        if first_idx > 0:
            prev_cue = cues[first_idx - 1]
            gap = cues[first_idx].start_ms - prev_cue.end_ms
            if gap <= 1500 and prev_cue.text and prev_cue.text[-1] not in ".!?\"'":
                if prev_cue.text not in context_parts:
                    context_parts.insert(0, prev_cue.text)

        if last_idx + 1 < len(cues):
            next_cue = cues[last_idx + 1]
            gap = next_cue.start_ms - cues[last_idx].end_ms
            if gap <= 1500 and active_text and active_text[-1] not in ".!?":
                if next_cue.text not in context_parts:
                    context_parts.append(next_cue.text)

        context_sentence = " ".join(context_parts).strip()
        return active_text, context_sentence
