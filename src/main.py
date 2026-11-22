"""Main system-tray application orchestrating MPC-HC polling, subtitles, dictionary, and overlay."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

# Allow running both `python src/main.py` and `python -m src.main`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.dictionary_service import DictionaryService, WordDefinition
from src.mpc_client import (
    MpcClient,
    MpcState,
    enable_mpc_web_interface,
    get_mpc_web_interface_status,
    get_mpc_window_rect,
    is_mpc_or_own_foreground,
)
from src.overlay_ui import OverlayWindow, SettingsDialog
from src.ocr_subtitle import capture_subtitle_region, recognize_image_text
from src.startup_helper import is_auto_start_enabled, set_auto_start
from src.subtitle_extractor import SubtitleExtractor, SubtitleLoadResult
from src.word_analyzer import CandidateWord, WordAnalyzer


def _create_tray_icon() -> QIcon:
    """Render a clean vector icon in memory so no external .ico file is needed."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    path = QPainterPath()
    path.addRoundedRect(4.0, 4.0, 56.0, 56.0, 14.0, 14.0)
    painter.fillPath(path, QColor(15, 23, 42))

    painter.setPen(QColor(56, 189, 248))
    painter.drawRoundedRect(5, 5, 54, 54, 13, 13)

    font = QFont("Segoe UI", 30, QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor(248, 250, 252))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "V")
    painter.end()
    return QIcon(pixmap)


class _WorkerSignals(QObject):
    poll_done = pyqtSignal(object)  # MpcState
    extract_done = pyqtSignal(str, object)  # filepath, SubtitleLoadResult
    gemini_done = pyqtSignal(int, object)  # generation_id, WordDefinition
    ocr_done = pyqtSignal(int, object, str)  # generation_id, pause_key, recognized_text


class _PollWorker(QRunnable):
    def __init__(self, client: MpcClient, signals: _WorkerSignals) -> None:
        super().__init__()
        self.client = client
        self.signals = signals

    def run(self) -> None:
        state = self.client.poll()
        self.signals.poll_done.emit(state)


class _ExtractWorker(QRunnable):
    def __init__(self, extractor: SubtitleExtractor, filepath: str, signals: _WorkerSignals) -> None:
        super().__init__()
        self.extractor = extractor
        self.filepath = filepath
        self.signals = signals

    def run(self) -> None:
        result = self.extractor.load_for_video(self.filepath)
        self.signals.extract_done.emit(self.filepath, result)


class _GeminiWorker(QRunnable):
    def __init__(
        self,
        dictionary: DictionaryService,
        generation_id: int,
        word: str,
        lemma: str,
        context_sentence: str,
        force_refresh: bool,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self.dictionary = dictionary
        self.generation_id = generation_id
        self.word = word
        self.lemma = lemma
        self.context_sentence = context_sentence
        self.force_refresh = force_refresh
        self.signals = signals

    def run(self) -> None:
        defn = self.dictionary.lookup_gemini(
            word=self.word,
            lemma=self.lemma,
            context_sentence=self.context_sentence,
            force_refresh=self.force_refresh,
        )
        self.signals.gemini_done.emit(self.generation_id, defn)


class _OcrWorker(QRunnable):
    def __init__(
        self,
        generation_id: int,
        pause_key: tuple[str, int],
        image,
        signals: _WorkerSignals,
    ) -> None:
        super().__init__()
        self.generation_id = generation_id
        self.pause_key = pause_key
        self.image = image
        self.signals = signals

    def run(self) -> None:
        try:
            text = recognize_image_text(self.image)
        except Exception:
            text = ""
        self.signals.ocr_done.emit(self.generation_id, self.pause_key, text)


class VocabAssistantController(QObject):
    """Coordinates MPC-HC polling, subtitle extraction, analysis, and overlay UI."""

    def __init__(self, app: QApplication, config: AppConfig) -> None:
        super().__init__()
        self.app = app
        self.config = config
        self.config.ensure_data_dirs()

        self.mpc_client = MpcClient(port=self.config.mpc_port)
        self.extractor = SubtitleExtractor(self.config.resolve_path(self.config.sub_cache_dir))
        self.analyzer = WordAnalyzer()
        self.dictionary = DictionaryService(self.config)

        self.overlay = OverlayWindow(
            initial_zipf=self.config.zipf_threshold,
            width=self.config.overlay_width,
        )
        self.overlay.dismiss_requested.connect(self._on_dismiss_overlay)
        self.overlay.rarity_changed.connect(self._on_rarity_changed)
        self.overlay.context_requested.connect(self._on_context_requested)
        self.overlay.save_requested.connect(self._on_save_requested)

        self.thread_pool = QThreadPool.globalInstance()
        self.signals = _WorkerSignals()
        self.signals.poll_done.connect(self._on_poll_done)
        self.signals.extract_done.connect(self._on_extract_done)
        self.signals.gemini_done.connect(self._on_gemini_done)
        self.signals.ocr_done.connect(self._on_ocr_done)

        self._assistant_enabled = True
        self._poll_in_flight = False
        self._extracting_paths: set[str] = set()
        self._last_state: MpcState = MpcState(connected=False)
        self._rendered_pause_key: Optional[tuple[str, int]] = None
        self._dismissed_pause_pos: Optional[tuple[str, int]] = None
        self._lookup_generation = 0
        self._ocr_generation = 0
        self._ocr_in_flight_key: Optional[tuple[str, int]] = None
        self._ocr_cache: dict[tuple[str, int], str] = {}

        self._current_candidates: list[CandidateWord] = []
        self._current_active_text: str = ""
        self._current_context: str = ""
        self._current_definitions: dict[str, WordDefinition] = {}

        self._config_save_timer = QTimer(self)
        self._config_save_timer.setSingleShot(True)
        self._config_save_timer.timeout.connect(self.config.save)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._tick_poll)
        self._poll_timer.start(self.config.poll_interval_ms)

        self._setup_tray()
        self._check_initial_integration()

    def _setup_tray(self) -> None:
        self.tray = QSystemTrayIcon(_create_tray_icon(), self)
        self.tray.setToolTip("MPC-HC Vocab Assistant")

        menu = QMenu()
        self.status_action = QAction("MPC-HC: Waiting...", menu)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)

        menu.addSeparator()

        self.toggle_action = QAction("Pause Assistant", menu)
        self.toggle_action.triggered.connect(self._on_toggle_assistant)
        menu.addAction(self.toggle_action)

        enable_mpc_action = QAction("Enable MPC-HC Integration", menu)
        enable_mpc_action.triggered.connect(self._on_enable_mpc_clicked)
        menu.addAction(enable_mpc_action)

        self.autostart_action = QAction("Start with Windows", menu)
        self.autostart_action.setCheckable(True)
        self.autostart_action.setChecked(is_auto_start_enabled())
        self.autostart_action.triggered.connect(self._on_toggle_autostart)
        menu.addAction(self.autostart_action)

        open_csv_action = QAction("Open Saved Vocab CSV", menu)
        open_csv_action.triggered.connect(self._on_open_csv)
        menu.addAction(open_csv_action)

        settings_action = QAction("Settings...", menu)
        settings_action.triggered.connect(self._on_open_settings)
        menu.addAction(settings_action)

        menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._on_quit)
        menu.addAction(quit_action)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _check_initial_integration(self) -> None:
        reg = get_mpc_web_interface_status()
        if reg.get("available") and not reg.get("enabled"):
            self.tray.showMessage(
                "MPC-HC Integration Needed",
                "Right-click this tray icon and click 'Enable MPC-HC Integration', "
                "then restart MPC-HC64.",
                QSystemTrayIcon.MessageIcon.Information,
                6000,
            )

    def _tick_poll(self) -> None:
        if not self._assistant_enabled or self._poll_in_flight:
            return
        self._poll_in_flight = True
        self.thread_pool.start(_PollWorker(self.mpc_client, self.signals))

    def _ensure_subtitle_loading(self, filepath: str) -> Optional[SubtitleLoadResult]:
        if not filepath:
            return None
        cached = self.extractor.get_cached_result(filepath)
        if cached is not None:
            return cached
        norm = os.path.normcase(os.path.abspath(filepath))
        if norm not in self._extracting_paths:
            self._extracting_paths.add(norm)
            self.thread_pool.start(_ExtractWorker(self.extractor, filepath, self.signals))
        return None

    def _on_extract_done(self, filepath: str, result: SubtitleLoadResult) -> None:
        norm = os.path.normcase(os.path.abspath(filepath))
        self._extracting_paths.discard(norm)
        if (
            self._assistant_enabled
            and self._last_state.is_paused
            and os.path.normcase(os.path.abspath(self._last_state.filepath or "")) == norm
        ):
            self._rendered_pause_key = None
            self._refresh_paused_overlay(self._last_state)

    def _on_poll_done(self, state: MpcState) -> None:
        self._poll_in_flight = False
        self._last_state = state

        if not self._assistant_enabled:
            self.overlay.hide()
            return

        if not state.connected:
            self.status_action.setText("MPC-HC: Not connected")
            self._hide_and_reset()
            return

        if state.filepath:
            self._ensure_subtitle_loading(state.filepath)

        if state.is_playing:
            self.status_action.setText(f"Playing: {state.filename or 'Video'}")
            self._dismissed_pause_pos = None
            self._rendered_pause_key = None
            if self.overlay.isVisible():
                self.overlay.hide()
            return

        if not state.is_paused or not state.filepath:
            self.status_action.setText("MPC-HC: Idle")
            self._hide_and_reset()
            return

        self.status_action.setText(f"Paused: {state.filename or 'Video'}")

        # Check if user dismissed overlay for this pause position
        if self._dismissed_pause_pos is not None:
            d_path, d_pos = self._dismissed_pause_pos
            if d_path == state.filepath and abs(state.position_ms - d_pos) <= 450:
                self.overlay.hide()
                return
            self._dismissed_pause_pos = None

        rect = get_mpc_window_rect(logical=True)
        own_hwnd = int(self.overlay.winId()) if self.overlay.isVisible() else None
        if rect is None or not is_mpc_or_own_foreground(own_hwnd):
            self.overlay.hide()
            return

        self.overlay.dock_to_rect(*rect)

        pause_key = (state.filepath, state.position_ms // 200)
        if pause_key != self._rendered_pause_key:
            self._rendered_pause_key = pause_key
            self._refresh_paused_overlay(state)

        if not self.overlay.isVisible():
            self.overlay.show()

    def _hide_and_reset(self) -> None:
        self._rendered_pause_key = None
        self._dismissed_pause_pos = None
        if self.overlay.isVisible():
            self.overlay.hide()

    def _refresh_paused_overlay(self, state: MpcState) -> None:
        sub_res = self._ensure_subtitle_loading(state.filepath)
        if sub_res is None:
            self._current_candidates = []
            self._current_active_text = ""
            self._current_context = ""
            self._current_definitions = {}
            self.overlay.set_content(
                "",
                [],
                status_message="Extracting embedded subtitles from video...",
                reset_saved_marks=True,
            )
            return

        if sub_res.status in ("image_only", "no_subtitles"):
            self._handle_ocr_fallback(state, sub_res)
            return

        if sub_res.status != "ready":
            self._clear_overlay_state(sub_res.message)
            return

        active_text, context_sentence = self.extractor.get_subtitle_at(
            state.filepath,
            state.position_ms,
            buffer_ms=self.config.subtitle_window_ms,
        )
        if not active_text:
            self._clear_overlay_state("No subtitle line at this timestamp.")
            return

        self._apply_subtitle_text(active_text, context_sentence or active_text)

    def _clear_overlay_state(self, status_message: str) -> None:
        self._current_candidates = []
        self._current_active_text = ""
        self._current_context = ""
        self._current_definitions = {}
        self.overlay.set_content(
            "",
            [],
            status_message=status_message,
            reset_saved_marks=True,
        )

    def _apply_subtitle_text(self, active_text: str, context_sentence: str) -> None:
        is_new_subtitle = active_text != self._current_active_text
        self._current_active_text = active_text
        self._current_context = context_sentence or active_text
        self._current_candidates = self.analyzer.extract_all_candidates(active_text)
        self._populate_from_candidates(is_new_subtitle=is_new_subtitle)

    def _handle_ocr_fallback(self, state: MpcState, sub_res: SubtitleLoadResult) -> None:
        pause_key = (state.filepath, state.position_ms // 200)
        if pause_key in self._ocr_cache:
            cached_text = self._ocr_cache[pause_key]
            if cached_text:
                self._apply_subtitle_text(cached_text, cached_text)
            else:
                self._clear_overlay_state("No subtitle visible on screen at this pause.")
            return

        phys_rect = get_mpc_window_rect(logical=False)
        if phys_rect is None:
            self._clear_overlay_state(sub_res.message)
            return

        if pause_key == self._ocr_in_flight_key:
            return

        try:
            img = capture_subtitle_region(phys_rect)
        except Exception as exc:
            self._clear_overlay_state(f"{sub_res.message}\nOCR capture error: {exc}")
            return

        self._ocr_generation += 1
        self._ocr_in_flight_key = pause_key
        self._clear_overlay_state("Reading on-screen subtitle (OCR)...")
        self.thread_pool.start(
            _OcrWorker(
                generation_id=self._ocr_generation,
                pause_key=pause_key,
                image=img,
                signals=self.signals,
            )
        )

    def _on_ocr_done(self, generation_id: int, pause_key: object, recognized_text: str) -> None:
        if isinstance(pause_key, tuple):
            if len(self._ocr_cache) >= 200:
                self._ocr_cache.clear()
            self._ocr_cache[pause_key] = recognized_text
        if self._ocr_in_flight_key == pause_key:
            self._ocr_in_flight_key = None

        if generation_id != self._ocr_generation:
            return
        if not self._assistant_enabled or not self._last_state.is_paused:
            return

        current_key = (self._last_state.filepath, self._last_state.position_ms // 200)
        if current_key != pause_key:
            return

        if recognized_text:
            self._apply_subtitle_text(recognized_text, recognized_text)
        else:
            self._clear_overlay_state("No subtitle visible on screen at this pause.")

    def _populate_from_candidates(self, is_new_subtitle: bool) -> None:
        self._lookup_generation += 1
        gen = self._lookup_generation

        filtered = self.analyzer.filter_candidates(
            self._current_candidates, zipf_threshold=self.config.zipf_threshold
        )
        definitions: list[WordDefinition] = []
        pending_gemini: list[CandidateWord] = []

        for cand in filtered:
            cached_ai = self.dictionary.get_cached_gemini(cand.surface, self._current_context)
            if cached_ai is not None:
                definitions.append(cached_ai)
                continue

            offline = self.dictionary.lookup_offline(cand.surface, cand.lemma, cand.pos_hint)
            if offline is not None:
                definitions.append(offline)
                continue

            if self.config.auto_llm_for_unknown:
                definitions.append(
                    WordDefinition(
                        word=cand.surface,
                        lemma=cand.lemma,
                        pos=cand.pos_hint,
                        definition="Asking Gemini for contextual meaning...",
                        source="loading",
                    )
                )
                pending_gemini.append(cand)
            else:
                definitions.append(
                    WordDefinition(
                        word=cand.surface,
                        lemma=cand.lemma,
                        pos=cand.pos_hint,
                        definition="Not in offline dictionary. Click '✨ Ask Gemini' for an AI explanation + example.",
                        source="not_found",
                    )
                )

        self._current_definitions = {d.word.lower(): d for d in definitions}
        self.overlay.set_content(
            self._current_active_text,
            definitions,
            reset_saved_marks=is_new_subtitle,
        )

        for cand in pending_gemini:
            self.thread_pool.start(
                _GeminiWorker(
                    dictionary=self.dictionary,
                    generation_id=gen,
                    word=cand.surface,
                    lemma=cand.lemma,
                    context_sentence=self._current_context,
                    force_refresh=False,
                    signals=self.signals,
                )
            )

    def _on_gemini_done(self, generation_id: int, defn: WordDefinition) -> None:
        if generation_id != self._lookup_generation:
            return
        self._current_definitions[defn.word.lower()] = defn
        self.overlay.update_definition(defn)

    def _on_dismiss_overlay(self) -> None:
        if self._last_state.filepath:
            self._dismissed_pause_pos = (self._last_state.filepath, self._last_state.position_ms)
        self.overlay.hide()

    def _on_rarity_changed(self, threshold: float) -> None:
        self.config.zipf_threshold = round(threshold, 1)
        self._config_save_timer.start(500)
        if self._current_active_text:
            self._populate_from_candidates(is_new_subtitle=False)

    def _on_context_requested(self, word: str, lemma: str) -> None:
        loading_defn = WordDefinition(
            word=word,
            lemma=lemma,
            pos="",
            definition="Asking Gemini for sentence-specific explanation...",
            source="loading",
        )
        self._current_definitions[word.lower()] = loading_defn
        self.overlay.update_definition(loading_defn)
        self.thread_pool.start(
            _GeminiWorker(
                dictionary=self.dictionary,
                generation_id=self._lookup_generation,
                word=word,
                lemma=lemma,
                context_sentence=self._current_context or self._current_active_text,
                force_refresh=False,
                signals=self.signals,
            )
        )

    def _on_save_requested(self, word: str) -> None:
        defn = self._current_definitions.get(word.lower())
        if defn is None:
            return
        self.dictionary.save_word_to_csv(
            defn=defn,
            sentence=self._current_context or self._current_active_text,
            filename=self._last_state.filename or Path(self._last_state.filepath).name,
            timestamp_ms=self._last_state.position_ms,
        )
        self.overlay.mark_word_saved(word)

    def _on_toggle_assistant(self) -> None:
        self._assistant_enabled = not self._assistant_enabled
        if self._assistant_enabled:
            self.toggle_action.setText("Pause Assistant")
            self.status_action.setText("MPC-HC: Polling...")
        else:
            self.toggle_action.setText("Resume Assistant")
            self.status_action.setText("Assistant Paused")
            self.overlay.hide()

    def _on_enable_mpc_clicked(self) -> None:
        ok, msg = enable_mpc_web_interface(port=self.config.mpc_port)
        if ok:
            QMessageBox.information(None, "MPC-HC Integration Enabled", msg)
        else:
            QMessageBox.warning(None, "MPC-HC Integration Error", msg)

    def _on_toggle_autostart(self, checked: bool) -> None:
        ok, msg = set_auto_start(checked)
        if not ok:
            self.autostart_action.setChecked(not checked)
            QMessageBox.warning(None, "Startup Setting Error", msg)
        else:
            self.tray.showMessage(
                "Startup Updated",
                msg,
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )

    def _on_open_csv(self) -> None:
        csv_path = self.config.resolve_path(self.config.vocab_csv_path)
        if not csv_path.exists():
            QMessageBox.information(
                None,
                "Saved Vocabulary",
                f"No saved words yet.\nClick '+ Save' on any word card to save to:\n{csv_path}",
            )
            return
        if sys.platform == "win32":
            os.startfile(str(csv_path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(csv_path)])

    def _on_open_settings(self) -> None:
        dialog = SettingsDialog(self.config)
        if dialog.exec():
            self.mpc_client.set_port(self.config.mpc_port)
            self.overlay.set_panel_width(self.config.overlay_width)
            self.overlay.set_slider_value(self.config.zipf_threshold)
            self._rendered_pause_key = None

    def _on_quit(self) -> None:
        self._poll_timer.stop()
        self.overlay.hide()
        self.tray.hide()
        self.mpc_client.close()
        self.app.quit()


def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("MPC-HC Vocab Assistant")

    config = AppConfig.load()
    _controller = VocabAssistantController(app, config)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
