"""Offline WordNet dictionary + Gemini API contextual explanation + CSV export."""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from nltk.corpus import wordnet

from .config import AppConfig

_WN_POS_MAP = {
    "noun": wordnet.NOUN,
    "verb": wordnet.VERB,
    "adj": wordnet.ADJ,
    "adv": wordnet.ADV,
}

_WN_POS_LABEL = {
    "n": "noun",
    "v": "verb",
    "a": "adj",
    "s": "adj",
    "r": "adv",
}

_GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "corrected_word": {
            "type": "string",
            "description": (
                "If the target word is an OCR or spelling typo, the intended English word inferred from "
                "sentence context and visual character similarity; otherwise empty string."
            ),
        },
        "pos": {
            "type": "string",
            "description": "Short part of speech, e.g. noun, verb, adj, adv, slang, idiom.",
        },
        "meaning": {
            "type": "string",
            "description": "Clear, concise explanation (under 30 words) of the word (or corrected word) in this exact sentence context.",
        },
        "example": {
            "type": "string",
            "description": "One short, natural example sentence showing how the word is used.",
        },
        "simple_synonym": {
            "type": "string",
            "description": "1-3 word plain-English synonym or short equivalent.",
        },
    },
    "required": ["corrected_word", "pos", "meaning", "example", "simple_synonym"],
}

_SYSTEM_INSTRUCTION = (
    "You are a concise, accurate English vocabulary assistant for someone watching videos with subtitles. "
    "Note that subtitles may come from screen OCR and can contain visual character misreads or typos "
    "(such as 'wpmer' for 'women', 'rn' for 'm', 'o'/'p', 'n'/'r', 'l'/'I'). "
    "If the target word is not a standard word or does not fit grammatically, first infer the intended word "
    "using the full subtitle sentence context and visual similarity, put that intended word in 'corrected_word', "
    "and explain that intended word; otherwise leave 'corrected_word' as an empty string. "
    "Provide the meaning specifically as used in the subtitle sentence, plus one short, natural usage example sentence. "
    "Keep the meaning clear, simple, and under 30 words."
)


@dataclass
class WordDefinition:
    """Resolved definition entry displayed in the overlay."""

    word: str
    lemma: str
    pos: str = ""
    definition: str = ""
    example: str = ""
    synonym: str = ""
    source: str = "offline"  # "offline" | "gemini" | "loading" | "not_found"


def _format_timestamp(ms: int) -> str:
    total_sec = max(0, int(ms // 1000))
    hrs, rem = divmod(total_sec, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


class DictionaryService:
    """Provides instant offline WordNet definitions and Gemini API contextual fallback."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._llm_cache: dict[str, dict[str, str]] = self._load_llm_cache()
        self._client: Optional[genai.Client] = None
        self._client_key: str = ""

    def _cache_file(self) -> Path:
        return self.config.resolve_path(self.config.llm_cache_path)

    def _load_llm_cache(self) -> dict[str, dict[str, str]]:
        path = self._cache_file()
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_llm_cache(self) -> None:
        path = self._cache_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._llm_cache, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _cache_key(word: str, context_sentence: str) -> str:
        return f"{word.strip().lower()}||{context_sentence.strip()}"

    def _get_genai_client(self, api_key: str) -> genai.Client:
        with self._lock:
            if self._client is None or self._client_key != api_key:
                self._client = genai.Client(api_key=api_key)
                self._client_key = api_key
            return self._client

    def lookup_offline(
        self, word: str, lemma: str = "", pos_hint: str = ""
    ) -> Optional[WordDefinition]:
        """Query local NLTK WordNet dictionary (<5ms)."""
        target_lemma = (lemma or word).strip().lower()
        surface = word.strip().lower()
        wn_pos = _WN_POS_MAP.get(pos_hint)

        synsets = []
        if wn_pos:
            synsets = wordnet.synsets(target_lemma, pos=wn_pos)
        if not synsets:
            synsets = wordnet.synsets(target_lemma)
        if not synsets and surface != target_lemma:
            if wn_pos:
                synsets = wordnet.synsets(surface, pos=wn_pos)
            if not synsets:
                synsets = wordnet.synsets(surface)

        if not synsets:
            return None

        best = synsets[0]
        pos_label = _WN_POS_LABEL.get(best.pos(), pos_hint or "")
        definition = best.definition().strip()
        if definition:
            definition = definition[0].upper() + definition[1:]
        examples = best.examples()
        example = examples[0].strip() if examples else ""

        synonym = ""
        for cand in best.lemma_names():
            clean_cand = cand.replace("_", " ").strip()
            if clean_cand.lower() not in {surface, target_lemma}:
                synonym = clean_cand
                break

        return WordDefinition(
            word=word,
            lemma=target_lemma,
            pos=pos_label,
            definition=definition,
            example=example,
            synonym=synonym,
            source="offline",
        )

    @staticmethod
    def _build_gemini_definition(word: str, entry: dict[str, str]) -> WordDefinition:
        corrected = entry.get("corrected_word", "").strip()
        lemma = entry.get("lemma", word.lower()).strip() or word.lower()
        meaning = entry.get("meaning", "").strip()

        if corrected and corrected.lower() != word.lower():
            lemma = corrected
            if not meaning.lower().startswith("(ocr typo for") and corrected.lower() not in meaning.lower():
                meaning = f"(OCR typo for '{corrected}') {meaning}"

        return WordDefinition(
            word=word,
            lemma=lemma,
            pos=entry.get("pos", "").strip(),
            definition=meaning,
            example=entry.get("example", "").strip(),
            synonym=entry.get("simple_synonym", "").strip(),
            source="gemini",
        )

    def get_cached_gemini(self, word: str, context_sentence: str) -> Optional[WordDefinition]:
        """Return cached Gemini explanation if already fetched for (word, context)."""
        key = self._cache_key(word, context_sentence)
        with self._lock:
            cached = self._llm_cache.get(key)
        if not cached:
            return None
        return self._build_gemini_definition(word, cached)

    def lookup_gemini(
        self,
        word: str,
        lemma: str = "",
        context_sentence: str = "",
        force_refresh: bool = False,
    ) -> WordDefinition:
        """Fetch context-aware explanation via Gemini Interactions API and cache it."""
        target_lemma = (lemma or word).strip().lower()
        ctx = (context_sentence or "").strip()

        if not force_refresh:
            cached = self.get_cached_gemini(word, ctx)
            if cached is not None:
                return cached

        api_key = self.config.get_effective_api_key()
        if not api_key:
            return WordDefinition(
                word=word,
                lemma=target_lemma,
                pos="",
                definition=(
                    "Not in offline dictionary. Add a Gemini API key in Settings "
                    "(or set GEMINI_API_KEY) for AI context explanations."
                ),
                source="not_found",
            )

        prompt = (
            f'Target word: "{word}" (base form: "{target_lemma}")\n'
            f'Subtitle sentence: "{ctx or word}"'
        )

        try:
            client = self._get_genai_client(api_key)
            interaction = client.interactions.create(
                model=self.config.gemini_model,
                system_instruction=_SYSTEM_INSTRUCTION,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": _GEMINI_SCHEMA,
                },
                store=False,
                generation_config={"max_output_tokens": 260},
            )
            raw_text = (interaction.output_text or "").strip()
            parsed = self._parse_gemini_json(raw_text)

            entry = {
                "lemma": target_lemma,
                "corrected_word": parsed.get("corrected_word", "").strip(),
                "pos": parsed.get("pos", "").strip(),
                "meaning": parsed.get("meaning", "").strip(),
                "example": parsed.get("example", "").strip(),
                "simple_synonym": parsed.get("simple_synonym", "").strip(),
            }
            if entry["meaning"]:
                key = self._cache_key(word, ctx)
                with self._lock:
                    self._llm_cache[key] = entry
                    self._save_llm_cache()
                return self._build_gemini_definition(word, entry)

            return WordDefinition(
                word=word,
                lemma=target_lemma,
                pos="",
                definition="Gemini returned an empty explanation.",
                source="not_found",
            )
        except Exception as exc:
            return WordDefinition(
                word=word,
                lemma=target_lemma,
                pos="",
                definition=f"Gemini lookup failed: {exc}",
                source="not_found",
            )

    @staticmethod
    def _parse_gemini_json(raw_text: str) -> dict[str, str]:
        if not raw_text:
            return {}
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items() if v is not None}
        except json.JSONDecodeError:
            pass
        # Fallback if text includes markdown fences
        cleaned = raw_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items() if v is not None}
        except json.JSONDecodeError:
            return {"pos": "", "meaning": raw_text, "simple_synonym": ""}
        return {}

    def save_word_to_csv(
        self,
        defn: WordDefinition,
        sentence: str,
        filename: str,
        timestamp_ms: int,
    ) -> Path:
        """Append a saved vocabulary word to the configured CSV file."""
        csv_path = self.config.resolve_path(self.config.vocab_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        with self._lock:
            with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(
                        [
                            "saved_at",
                            "word",
                            "lemma",
                            "pos",
                            "definition",
                            "synonym",
                            "source",
                            "subtitle_sentence",
                            "video_file",
                            "timestamp",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        defn.word,
                        defn.lemma,
                        defn.pos,
                        defn.definition,
                        defn.synonym,
                        defn.source,
                        sentence,
                        filename,
                        _format_timestamp(timestamp_ms),
                    ]
                )
        return csv_path
