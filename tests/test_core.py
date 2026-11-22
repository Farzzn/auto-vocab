"""Comprehensive verification tests for MPC-HC Vocabulary Assistant."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image, ImageDraw, ImageFont
from PyQt6.QtWidgets import QApplication

from src.config import AppConfig
from src.dictionary_service import DictionaryService, WordDefinition
from src.mpc_client import parse_variables_html
from src.ocr_subtitle import recognize_image_text
from src.overlay_ui import OverlayWindow
from src.subtitle_extractor import SubtitleExtractor, clean_subtitle_text
from src.word_analyzer import WordAnalyzer


class TestMpcClientParsing(unittest.TestCase):
    def test_parse_variables_html_paused(self) -> None:
        sample_html = """
        <!DOCTYPE html>
        <html>
        <body>
            <p id="filepath">D:\\Videos\\Sample Movie.mkv</p>
            <p id="file">Sample Movie.mkv</p>
            <p id="state">1</p>
            <p id="statestring">Paused</p>
            <p id="position">12345</p>
            <p id="duration">900000</p>
        </body>
        </html>
        """
        st = parse_variables_html(sample_html)
        self.assertTrue(st.connected)
        self.assertTrue(st.is_paused)
        self.assertFalse(st.is_playing)
        self.assertEqual(st.state, 1)
        self.assertEqual(st.filepath, r"D:\Videos\Sample Movie.mkv")
        self.assertEqual(st.filename, "Sample Movie.mkv")
        self.assertEqual(st.position_ms, 12345)
        self.assertEqual(st.duration_ms, 900000)


class TestSubtitleExtractor(unittest.TestCase):
    def test_clean_subtitle_text(self) -> None:
        raw = r"{\an8}<i>He exacerbated</i>\Nthe <b>cacophony</b>."
        self.assertEqual(clean_subtitle_text(raw), "He exacerbated the cacophony.")

    def test_extract_embedded_mkv_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            srt_src = tmp_path / "source_track.srt"
            srt_src.write_text(
                "1\n"
                "00:00:00,200 --> 00:00:01,500\n"
                "The ephemeral glow faded quickly.\n\n"
                "2\n"
                "00:00:01,800 --> 00:00:02,900\n"
                "His nonchalant attitude was uncanny.\n",
                encoding="utf-8",
            )
            mkv_path = tmp_path / "sample.mkv"
            cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=320x240:d=3",
                "-i",
                str(srt_src),
                "-c:v",
                "libx264",
                "-t",
                "3",
                "-c:s",
                "srt",
                "-metadata:s:s:0",
                "language=eng",
                str(mkv_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, msg=f"ffmpeg setup failed: {proc.stderr}")

            cache_dir = tmp_path / "sub_cache"
            extractor = SubtitleExtractor(cache_dir)
            result = extractor.load_for_video(str(mkv_path))
            self.assertEqual(result.status, "ready", msg=result.message)
            self.assertEqual(result.source_kind, "embedded")
            self.assertEqual(len(result.cues), 2)

            active1, ctx1 = extractor.get_subtitle_at(str(mkv_path), 800, buffer_ms=500)
            self.assertEqual(active1, "The ephemeral glow faded quickly.")
            self.assertIn("ephemeral", ctx1)

            # Test buffer just after cue 1 ends (1700ms is 200ms after 1500ms)
            active_buf, _ = extractor.get_subtitle_at(str(mkv_path), 1700, buffer_ms=500)
            self.assertEqual(active_buf, "The ephemeral glow faded quickly.")

            active2, _ = extractor.get_subtitle_at(str(mkv_path), 2200, buffer_ms=500)
            self.assertEqual(active2, "His nonchalant attitude was uncanny.")


class TestWordAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = WordAnalyzer()

    def test_filters_stopwords_and_lemmatizes(self) -> None:
        text = "The politicians were blatantly obfuscating the ephemeral truth."
        all_cands = self.analyzer.extract_all_candidates(text)
        lemmas = {c.lemma for c in all_cands}

        self.assertNotIn("the", lemmas)
        self.assertNotIn("were", lemmas)
        self.assertIn("obfuscate", lemmas)
        self.assertIn("ephemeral", lemmas)

        # Strict rarity threshold should retain rare words and exclude common ones like "truth"
        rare = self.analyzer.filter_candidates(all_cands, zipf_threshold=4.2)
        rare_lemmas = {c.lemma for c in rare}
        self.assertIn("obfuscate", rare_lemmas)
        self.assertIn("ephemeral", rare_lemmas)
        self.assertNotIn("truth", rare_lemmas)

        # Raising threshold includes more words
        relaxed = self.analyzer.filter_candidates(all_cands, zipf_threshold=5.5)
        self.assertGreaterEqual(len(relaxed), len(rare))


class TestDictionaryService(unittest.TestCase):
    def _make_config(self, tmp_path: Path, api_key: str = "test-key") -> AppConfig:
        cfg = AppConfig(
            gemini_api_key=api_key,
            vocab_csv_path=str(tmp_path / "saved.csv"),
            llm_cache_path=str(tmp_path / "llm_cache.json"),
            sub_cache_dir=str(tmp_path / "sub_cache"),
            _config_path=tmp_path / "config.json",
        )
        cfg.ensure_data_dirs()
        return cfg

    def test_offline_wordnet_lookup(self) -> None:
        self.assertFalse(AppConfig().auto_llm_for_unknown)
        with tempfile.TemporaryDirectory() as tmp:
            svc = DictionaryService(self._make_config(Path(tmp)))
            defn = svc.lookup_offline("ephemeral", "ephemeral", "adj")
            self.assertIsNotNone(defn)
            assert defn is not None
            self.assertEqual(defn.source, "offline")
            self.assertTrue(len(defn.definition) > 5)

            # Unknown slang word should miss offline dictionary
            miss = svc.lookup_offline("skibidi", "skibidi", "noun")
            self.assertIsNone(miss)

    def test_gemini_fallback_mocked_and_cached_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path, api_key="fake-key")
            svc = DictionaryService(cfg)

            fake_json = json.dumps(
                {
                    "corrected_word": "",
                    "pos": "slang",
                    "meaning": "Having effortless charm or attractiveness.",
                    "example": "He walked in with effortless rizz and charmed everyone.",
                    "simple_synonym": "charisma",
                }
            )
            fake_interaction = SimpleNamespace(output_text=fake_json)
            fake_client = MagicMock()
            fake_client.interactions.create.return_value = fake_interaction

            with patch("src.dictionary_service.genai.Client", return_value=fake_client):
                res = svc.lookup_gemini(
                    word="rizz",
                    lemma="rizz",
                    context_sentence="That guy has unspoken rizz.",
                )
                self.assertEqual(res.source, "gemini")
                self.assertEqual(res.pos, "slang")
                self.assertEqual(res.synonym, "charisma")
                self.assertIn("charm", res.definition.lower())
                self.assertIn("rizz", res.example.lower())

                # Verify SDK call contract
                call_kwargs = fake_client.interactions.create.call_args.kwargs
                self.assertEqual(call_kwargs["model"], cfg.gemini_model)
                self.assertFalse(call_kwargs["store"])
                self.assertIn("system_instruction", call_kwargs)
                self.assertIn("response_format", call_kwargs)

                # Second call hits disk/memory cache without calling API again
                res2 = svc.lookup_gemini(
                    word="rizz",
                    lemma="rizz",
                    context_sentence="That guy has unspoken rizz.",
                )
                self.assertEqual(res2.source, "gemini")
                self.assertEqual(fake_client.interactions.create.call_count, 1)

            # Verify CSV saving
            csv_path = svc.save_word_to_csv(
                res,
                sentence="That guy has unspoken rizz.",
                filename="demo.mkv",
                timestamp_ms=65000,
            )
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1][1], "rizz")
            self.assertEqual(rows[1][9], "00:01:05")

    def test_gemini_typo_inference_mocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = self._make_config(tmp_path, api_key="fake-key")
            svc = DictionaryService(cfg)

            fake_json = json.dumps(
                {
                    "corrected_word": "women",
                    "pos": "noun",
                    "meaning": "Adult female human beings.",
                    "example": "Women often bring different perspectives to the conversation.",
                    "simple_synonym": "females",
                }
            )
            fake_client = MagicMock()
            fake_client.interactions.create.return_value = SimpleNamespace(output_text=fake_json)

            with patch("src.dictionary_service.genai.Client", return_value=fake_client):
                res = svc.lookup_gemini(
                    word="wpmer",
                    lemma="wpmer",
                    context_sentence="Sometimes with wpmer you wanna listen to what upsets them...",
                )
                self.assertEqual(res.source, "gemini")
                self.assertEqual(res.lemma, "women")
                self.assertIn("OCR typo for 'women'", res.definition)

    def test_gemini_live_smoke_if_key_present(self) -> None:
        cfg = AppConfig.load()
        api_key = cfg.get_effective_api_key()
        if not api_key:
            self.skipTest("GEMINI_API_KEY not configured in environment")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            live_cfg = self._make_config(tmp_path, api_key=api_key)
            svc = DictionaryService(live_cfg)
            res = svc.lookup_gemini(
                word="rizz",
                lemma="rizz",
                context_sentence="Everyone joked about his unspoken rizz at the party.",
            )
            self.assertEqual(res.source, "gemini", msg=f"Live Gemini failed: {res.definition}")
            self.assertTrue(len(res.definition) > 5)
            self.assertTrue(len(res.example) > 5)

            typo_res = svc.lookup_gemini(
                word="wpmer",
                lemma="wpmer",
                context_sentence="Sometimes with wpmer you wanna listen to what upsets them...",
            )
            self.assertEqual(typo_res.source, "gemini", msg=f"Live typo check failed: {typo_res.definition}")
            self.assertTrue(
                typo_res.lemma.lower() == "women" or "women" in typo_res.definition.lower(),
                msg=f"Expected Gemini to infer 'women', got lemma={typo_res.lemma!r}, def={typo_res.definition!r}",
            )


class TestOverlayUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_overlay_populates_and_updates_cards(self) -> None:
        overlay = OverlayWindow(initial_zipf=4.8, width=370)
        overlay.dock_to_rect(0, 0, 1920, 1080)
        self.assertLess(overlay.geometry().bottom(), int(1080 * 0.72))

        d1 = WordDefinition(
            word="ephemeral",
            lemma="ephemeral",
            pos="adj",
            definition="Lasting for a very short time.",
            example="fashions are ephemeral",
            synonym="short-lived",
            source="offline",
        )
        d2 = WordDefinition(
            word="rizz",
            lemma="rizz",
            pos="noun",
            definition="Not in offline dictionary. Click '✨ Ask Gemini' for an AI explanation + example.",
            source="not_found",
        )
        overlay.set_content("The ephemeral glow and unspoken rizz.", [d1, d2], reset_saved_marks=True)
        self.assertEqual(overlay.count_label.text(), "2 words")
        self.assertIn("ephemeral", overlay._cards)
        self.assertIn("rizz", overlay._cards)
        self.assertEqual(overlay._cards["rizz"].context_btn.text(), "✨ Ask Gemini")
        self.assertEqual(overlay._cards["rizz"].badge_label.text(), "Offline miss")

        # Update card in place when Gemini returns
        d2_done = WordDefinition(
            word="rizz",
            lemma="rizz",
            pos="slang",
            definition="Natural charm or appeal.",
            example="He won the room over with sheer rizz.",
            synonym="charisma",
            source="gemini",
        )
        overlay.update_definition(d2_done)
        self.assertEqual(overlay._cards["rizz"].badge_label.text(), "Gemini AI")
        self.assertIn("e.g.", overlay._cards["rizz"].meta_label.text())

        # Mark saved
        overlay.mark_word_saved("ephemeral")
        self.assertEqual(overlay._cards["ephemeral"].save_btn.text(), "✓ Saved")

        # Verify slider signal
        received: list[float] = []
        overlay.rarity_changed.connect(received.append)
        overlay.rarity_slider.setValue(42)
        self.assertEqual(received, [4.2])


class TestOcrSubtitle(unittest.TestCase):
    def test_recognize_synthetic_subtitle_image(self) -> None:
        img = Image.new("RGB", (900, 160), color=(18, 20, 26))
        draw = ImageDraw.Draw(img)
        font_path = Path(r"C:\Windows\Fonts\arial.ttf")
        if font_path.is_file():
            font = ImageFont.truetype(str(font_path), 36)
        else:
            font = ImageFont.load_default(size=36)
        draw.text((40, 50), "His ephemeral charm was uncanny.", fill=(255, 255, 255), font=font)

        recognized = recognize_image_text(img)
        lowered = recognized.lower()
        self.assertIn("ephemeral", lowered, msg=f"OCR output: {recognized!r}")
        self.assertIn("uncanny", lowered, msg=f"OCR output: {recognized!r}")


if __name__ == "__main__":
    unittest.main()
