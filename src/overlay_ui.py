"""Frameless non-focus-stealing PyQt6 overlay panel and Settings dialog."""

from __future__ import annotations

import ctypes
import sys
from typing import Iterable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig
from .dictionary_service import WordDefinition
from .mpc_client import enable_mpc_web_interface, get_mpc_web_interface_status

_GWL_EXSTYLE = -20
_WS_EX_NOACTIVATE = 0x08000000


def _apply_win32_noactivate(widget: QWidget) -> None:
    """Prevent the overlay window from stealing keyboard focus on Windows."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        user32 = ctypes.windll.user32
        get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
        set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
        ex_style = get_long(hwnd, _GWL_EXSTYLE)
        set_long(hwnd, _GWL_EXSTYLE, ex_style | _WS_EX_NOACTIVATE)
    except Exception:
        pass


class WordCard(QFrame):
    """Displays one vocabulary word with its definition and action buttons."""

    context_clicked = pyqtSignal(str, str)  # word, lemma
    save_clicked = pyqtSignal(str)  # word

    def __init__(self, defn: WordDefinition, is_saved: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.defn = defn
        self.is_saved = is_saved
        self.setObjectName("WordCard")
        self.setStyleSheet(
            """
            QFrame#WordCard {
                background-color: rgba(255, 255, 255, 14);
                border: 1px solid rgba(255, 255, 255, 24);
                border-radius: 10px;
            }
            """
        )
        self._build_ui()
        self.update_definition(defn, is_saved=is_saved)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(11, 9, 11, 9)
        root.setSpacing(5)

        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        self.word_label = QLabel()
        self.word_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.word_label.setStyleSheet(
            "color: #F8FAFC; font-size: 14px; font-weight: 700; background: transparent;"
        )
        top_row.addWidget(self.word_label)

        self.pos_label = QLabel()
        self.pos_label.setStyleSheet(
            "color: #94A3B8; font-size: 11px; font-style: italic; background: transparent;"
        )
        top_row.addWidget(self.pos_label)

        top_row.addStretch(1)

        self.badge_label = QLabel()
        top_row.addWidget(self.badge_label)
        root.addLayout(top_row)

        self.def_label = QLabel()
        self.def_label.setWordWrap(True)
        self.def_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.def_label.setStyleSheet(
            "color: #E2E8F0; font-size: 12.5px; line-height: 1.35; background: transparent;"
        )
        root.addWidget(self.def_label)

        self.meta_label = QLabel()
        self.meta_label.setWordWrap(True)
        self.meta_label.setStyleSheet(
            "color: #94A3B8; font-size: 11px; background: transparent;"
        )
        root.addWidget(self.meta_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        btn_row.addStretch(1)

        self.context_btn = QPushButton("✨ Context")
        self.context_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.context_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.context_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(139, 92, 246, 45);
                color: #DDD6FE;
                border: 1px solid rgba(139, 92, 246, 95);
                border-radius: 6px;
                padding: 3px 8px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(139, 92, 246, 80);
            }
            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 10);
                color: #64748B;
                border-color: rgba(255, 255, 255, 18);
            }
            """
        )
        self.context_btn.clicked.connect(
            lambda: self.context_clicked.emit(self.defn.word, self.defn.lemma)
        )
        btn_row.addWidget(self.context_btn)

        self.save_btn = QPushButton("+ Save")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save_btn.clicked.connect(lambda: self.save_clicked.emit(self.defn.word))
        btn_row.addWidget(self.save_btn)

        root.addLayout(btn_row)

    def update_definition(self, defn: WordDefinition, is_saved: Optional[bool] = None) -> None:
        self.defn = defn
        if is_saved is not None:
            self.is_saved = is_saved

        title = defn.word
        if defn.lemma and defn.lemma.lower() != defn.word.lower():
            title = f"{defn.word} ({defn.lemma})"
        self.word_label.setText(title)

        if defn.pos:
            self.pos_label.setText(defn.pos)
            self.pos_label.show()
        else:
            self.pos_label.hide()

        # Badge styling by source
        if defn.source == "offline":
            self.badge_label.setText("Offline")
            self.badge_label.setStyleSheet(
                "background-color: rgba(16, 185, 129, 38); color: #6EE7B7;"
                "border: 1px solid rgba(16, 185, 129, 80); border-radius: 5px;"
                "padding: 1px 6px; font-size: 10px; font-weight: 600;"
            )
            self.context_btn.setText("✨ Context")
            self.context_btn.setEnabled(True)
        elif defn.source == "gemini":
            self.badge_label.setText("Gemini AI")
            self.badge_label.setStyleSheet(
                "background-color: rgba(139, 92, 246, 50); color: #DDD6FE;"
                "border: 1px solid rgba(139, 92, 246, 110); border-radius: 5px;"
                "padding: 1px 6px; font-size: 10px; font-weight: 600;"
            )
            self.context_btn.setText("✨ Contextual")
            self.context_btn.setEnabled(False)
        elif defn.source == "loading":
            self.badge_label.setText("Asking Gemini...")
            self.badge_label.setStyleSheet(
                "background-color: rgba(245, 158, 11, 40); color: #FCD34D;"
                "border: 1px solid rgba(245, 158, 11, 90); border-radius: 5px;"
                "padding: 1px 6px; font-size: 10px; font-weight: 600;"
            )
            self.context_btn.setText("...")
            self.context_btn.setEnabled(False)
        else:
            self.badge_label.setText("Offline miss")
            self.badge_label.setStyleSheet(
                "background-color: rgba(245, 158, 11, 35); color: #FCD34D;"
                "border: 1px solid rgba(245, 158, 11, 80); border-radius: 5px;"
                "padding: 1px 6px; font-size: 10px; font-weight: 600;"
            )
            self.context_btn.setText("✨ Ask Gemini")
            self.context_btn.setEnabled(True)

        self.def_label.setText(defn.definition or "Looking up contextual meaning...")

        meta_parts: list[str] = []
        if defn.synonym:
            meta_parts.append(f"Syn: {defn.synonym}")
        if defn.example:
            meta_parts.append(f'e.g. "{defn.example}"')
        if meta_parts:
            self.meta_label.setText("  •  ".join(meta_parts))
            self.meta_label.show()
        else:
            self.meta_label.hide()

        self._refresh_save_button()

    def _refresh_save_button(self) -> None:
        if self.is_saved:
            self.save_btn.setText("✓ Saved")
            self.save_btn.setEnabled(False)
            self.save_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(16, 185, 129, 45);
                    color: #6EE7B7;
                    border: 1px solid rgba(16, 185, 129, 95);
                    border-radius: 6px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                }
                """
            )
        else:
            self.save_btn.setText("+ Save")
            self.save_btn.setEnabled(self.defn.source in {"offline", "gemini"})
            self.save_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(56, 189, 248, 35);
                    color: #BAE6FD;
                    border: 1px solid rgba(56, 189, 248, 85);
                    border-radius: 6px;
                    padding: 3px 8px;
                    font-size: 11px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(56, 189, 248, 70);
                }
                QPushButton:disabled {
                    background-color: rgba(255, 255, 255, 10);
                    color: #64748B;
                    border-color: rgba(255, 255, 255, 18);
                }
                """
            )


class OverlayWindow(QWidget):
    """Translucent right-side overlay docked to the paused MPC-HC window."""

    dismiss_requested = pyqtSignal()
    rarity_changed = pyqtSignal(float)
    context_requested = pyqtSignal(str, str)  # word, lemma
    save_requested = pyqtSignal(str)  # word

    def __init__(self, initial_zipf: float = 4.8, width: int = 370) -> None:
        super().__init__(None)
        self._panel_width = max(300, width)
        self._cards: dict[str, WordCard] = {}
        self._saved_words: set[str] = set()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self._build_ui(initial_zipf)

    def _build_ui(self, initial_zipf: float) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)

        title = QLabel("Vocab Assistant")
        title.setStyleSheet(
            "color: #F8FAFC; font-size: 14px; font-weight: 700; letter-spacing: 0.3px;"
        )
        header.addWidget(title)

        self.count_label = QLabel("0 words")
        self.count_label.setStyleSheet(
            "background-color: rgba(56, 189, 248, 35); color: #7DD3FC;"
            "border-radius: 9px; padding: 1px 8px; font-size: 11px; font-weight: 600;"
        )
        header.addWidget(self.count_label)
        header.addStretch(1)

        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        close_btn.setToolTip("Dismiss for this pause")
        close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255, 255, 255, 16);
                color: #CBD5E1;
                border: none;
                border-radius: 12px;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: rgba(244, 63, 94, 110);
                color: #FFFFFF;
            }
            """
        )
        close_btn.clicked.connect(self.dismiss_requested.emit)
        header.addWidget(close_btn)
        outer.addLayout(header)

        # Subtitle quote box
        self.subtitle_label = QLabel()
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.subtitle_label.setStyleSheet(
            """
            QLabel {
                background-color: rgba(255, 255, 255, 10);
                border-left: 3px solid #38BDF8;
                border-radius: 6px;
                padding: 7px 9px;
                color: #E2E8F0;
                font-size: 12px;
                font-style: italic;
            }
            """
        )
        outer.addWidget(self.subtitle_label)

        # Rarity filter slider
        slider_box = QFrame()
        slider_box.setStyleSheet(
            "background-color: rgba(255, 255, 255, 8); border-radius: 8px;"
        )
        slider_layout = QVBoxLayout(slider_box)
        slider_layout.setContentsMargins(9, 6, 9, 6)
        slider_layout.setSpacing(3)

        slider_top = QHBoxLayout()
        rarity_lbl = QLabel("Word Rarity Filter")
        rarity_lbl.setStyleSheet("color: #CBD5E1; font-size: 11px; font-weight: 600;")
        slider_top.addWidget(rarity_lbl)
        slider_top.addStretch(1)

        self.rarity_value_label = QLabel(f"Zipf < {initial_zipf:.1f}")
        self.rarity_value_label.setStyleSheet("color: #7DD3FC; font-size: 11px; font-weight: 600;")
        slider_top.addWidget(self.rarity_value_label)
        slider_layout.addLayout(slider_top)

        self.rarity_slider = QSlider(Qt.Orientation.Horizontal)
        self.rarity_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.rarity_slider.setMinimum(35)
        self.rarity_slider.setMaximum(60)
        self.rarity_slider.setValue(int(round(initial_zipf * 10)))
        self.rarity_slider.setStyleSheet(
            """
            QSlider::groove:horizontal {
                height: 4px;
                background: rgba(255, 255, 255, 35);
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #38BDF8;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #F8FAFC;
                width: 13px;
                height: 13px;
                margin: -5px 0;
                border-radius: 6px;
            }
            """
        )
        self.rarity_slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.rarity_slider)

        hints = QHBoxLayout()
        h_left = QLabel("Rare only")
        h_left.setStyleSheet("color: #64748B; font-size: 10px;")
        h_right = QLabel("More words")
        h_right.setStyleSheet("color: #64748B; font-size: 10px;")
        hints.addWidget(h_left)
        hints.addStretch(1)
        hints.addWidget(h_right)
        slider_layout.addLayout(hints)

        outer.addWidget(slider_box)

        # Status / empty banner
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #94A3B8; font-size: 12px; padding: 8px; background: transparent;"
        )
        self.status_label.hide()
        outer.addWidget(self.status_label)

        # Scroll area for cards
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet(
            """
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
                margin: 2px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 50);
                border-radius: 3px;
                min-height: 24px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            """
        )

        self.cards_container = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 2, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch(1)

        self.scroll.setWidget(self.cards_container)
        outer.addWidget(self.scroll, stretch=1)

    def _on_slider_changed(self, value: int) -> None:
        threshold = value / 10.0
        self.rarity_value_label.setText(f"Zipf < {threshold:.1f}")
        self.rarity_changed.emit(threshold)

    def set_slider_value(self, threshold: float) -> None:
        self.rarity_slider.blockSignals(True)
        self.rarity_slider.setValue(max(35, min(60, int(round(threshold * 10)))))
        self.rarity_value_label.setText(f"Zipf < {threshold:.1f}")
        self.rarity_slider.blockSignals(False)

    def set_panel_width(self, width: int) -> None:
        self._panel_width = max(300, width)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        _apply_win32_noactivate(self)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height()), 14.0, 14.0)
        painter.fillPath(path, QColor(15, 18, 28, 224))
        painter.setPen(QPen(QColor(255, 255, 255, 36), 1.0))
        painter.drawPath(path)

    def dock_to_rect(self, left: int, top: int, right: int, bottom: int) -> None:
        """Position the overlay on the right side of the MPC-HC frame above subtitles."""
        win_w = max(1, right - left)
        win_h = max(1, bottom - top)
        width = min(self._panel_width, max(280, int(win_w * 0.42)))
        top_pad = 48 if win_h > 300 else 16
        if win_h > 420:
            bottom_pad = max(150, int(win_h * 0.28))
        elif win_h > 300:
            bottom_pad = 64
        else:
            bottom_pad = 16
        avail_h = max(200, win_h - top_pad - bottom_pad)
        height = min(max(300, int(win_h * 0.66)), avail_h)
        x = right - width - 16
        y = top + top_pad
        self.setGeometry(x, y, width, height)

    def _clear_cards(self) -> None:
        self._cards.clear()
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def set_content(
        self,
        subtitle_text: str,
        definitions: Iterable[WordDefinition],
        status_message: str = "",
        reset_saved_marks: bool = False,
    ) -> None:
        """Populate the overlay panel for the current paused subtitle."""
        if reset_saved_marks:
            self._saved_words.clear()

        if subtitle_text:
            self.subtitle_label.setText(f'"{subtitle_text}"')
            self.subtitle_label.show()
        else:
            self.subtitle_label.hide()

        self._clear_cards()
        def_list = list(definitions)
        self.count_label.setText(f"{len(def_list)} word{'s' if len(def_list) != 1 else ''}")

        if status_message and not def_list:
            self.status_label.setText(status_message)
            self.status_label.show()
        elif not def_list:
            self.status_label.setText(
                "All words in this subtitle are common.\nDrag the rarity slider right to include more words."
            )
            self.status_label.show()
        else:
            self.status_label.hide()

        for defn in def_list:
            key = defn.word.lower()
            card = WordCard(defn, is_saved=(key in self._saved_words))
            card.context_clicked.connect(self.context_requested.emit)
            card.save_clicked.connect(self.save_requested.emit)
            self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
            self._cards[key] = card

    def update_definition(self, defn: WordDefinition) -> None:
        """Update an existing word card in place (e.g., when Gemini finishes)."""
        card = self._cards.get(defn.word.lower())
        if card is not None:
            card.update_definition(defn)

    def mark_word_saved(self, word: str) -> None:
        key = word.lower()
        self._saved_words.add(key)
        card = self._cards.get(key)
        if card is not None:
            card.update_definition(card.defn, is_saved=True)


class SettingsDialog(QDialog):
    """Configuration dialog for Gemini API, rarity threshold, and MPC-HC integration."""

    def __init__(self, config: AppConfig, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("MPC-HC Vocab Assistant Settings")
        self.setMinimumWidth(460)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.api_key_edit = QLineEdit(self.config.gemini_api_key)
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("Leave blank to use GEMINI_API_KEY env var")
        form.addRow("Gemini API Key:", self.api_key_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        models = [
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-4-hydrogen",
            "gemini-4-argon",
            "gemini-4-neon",
        ]
        self.model_combo.addItems(models)
        if self.config.gemini_model not in models:
            self.model_combo.addItem(self.config.gemini_model)
        self.model_combo.setCurrentText(self.config.gemini_model)
        form.addRow("Gemini Model:", self.model_combo)

        self.zipf_spin = QDoubleSpinBox()
        self.zipf_spin.setRange(3.5, 6.0)
        self.zipf_spin.setSingleStep(0.1)
        self.zipf_spin.setDecimals(1)
        self.zipf_spin.setValue(self.config.zipf_threshold)
        form.addRow("Default Rarity (Zipf <):", self.zipf_spin)

        self.auto_llm_check = QCheckBox("Automatically ask Gemini when offline dictionary misses a word")
        self.auto_llm_check.setChecked(self.config.auto_llm_for_unknown)
        form.addRow("", self.auto_llm_check)

        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(self.config.mpc_port)
        form.addRow("MPC-HC Web Port:", self.port_spin)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(300, 620)
        self.width_spin.setSingleStep(10)
        self.width_spin.setValue(self.config.overlay_width)
        form.addRow("Overlay Width (px):", self.width_spin)

        layout.addLayout(form)

        # MPC-HC Registry status & quick-enable button
        reg = get_mpc_web_interface_status()
        status_txt = (
            f"MPC-HC Integration: {'Enabled' if reg.get('enabled') else 'Disabled'} "
            f"(Port {reg.get('port')})"
        )
        self.mpc_status_label = QLabel(status_txt)
        layout.addWidget(self.mpc_status_label)

        enable_btn = QPushButton("Enable MPC-HC Integration Now")
        enable_btn.clicked.connect(self._on_enable_mpc)
        layout.addWidget(enable_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_enable_mpc(self) -> None:
        ok, msg = enable_mpc_web_interface(port=int(self.port_spin.value()))
        if ok:
            self.mpc_status_label.setText(f"MPC-HC Integration: Enabled (Port {self.port_spin.value()})")
            QMessageBox.information(self, "MPC-HC Integration", msg)
        else:
            QMessageBox.warning(self, "MPC-HC Integration", msg)

    def _on_save(self) -> None:
        self.config.gemini_api_key = self.api_key_edit.text().strip()
        self.config.gemini_model = self.model_combo.currentText().strip() or "gemini-3.5-flash-lite"
        self.config.zipf_threshold = round(float(self.zipf_spin.value()), 1)
        self.config.auto_llm_for_unknown = self.auto_llm_check.isChecked()
        self.config.mpc_port = int(self.port_spin.value())
        self.config.overlay_width = int(self.width_spin.value())
        self.config.save()
        self.accept()
