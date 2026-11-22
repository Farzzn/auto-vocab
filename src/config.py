"""Configuration management for MPC-HC Vocabulary Assistant."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = DATA_DIR / "config.json"


@dataclass
class AppConfig:
    """Application settings persisted to data/config.json."""

    mpc_port: int = 13579
    poll_interval_ms: int = 250
    zipf_threshold: float = 4.8
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: str = ""
    auto_llm_for_unknown: bool = False
    subtitle_window_ms: int = 500
    overlay_width: int = 370
    vocab_csv_path: str = "data/saved_vocab.csv"
    llm_cache_path: str = "data/llm_cache.json"
    sub_cache_dir: str = "data/sub_cache"
    _config_path: Path = field(default=DEFAULT_CONFIG_PATH, repr=False, compare=False)

    def resolve_path(self, rel_or_abs: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    def get_effective_api_key(self) -> str:
        """Return configured API key, falling back to GEMINI_API_KEY env var."""
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        key = (self.gemini_api_key or "").strip()
        if key:
            return key
        return (os.environ.get("GEMINI_API_KEY") or "").strip()

    def ensure_data_dirs(self) -> None:
        """Ensure required data and cache directories exist."""
        self.resolve_path(self.sub_cache_dir).mkdir(parents=True, exist_ok=True)
        self.resolve_path(self.llm_cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.resolve_path(self.vocab_csv_path).parent.mkdir(parents=True, exist_ok=True)
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Persist configuration to JSON."""
        self.ensure_data_dirs()
        data = asdict(self)
        data.pop("_config_path", None)
        self._config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> AppConfig:
        """Load configuration from disk, creating defaults if missing."""
        cfg_path = path or DEFAULT_CONFIG_PATH
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        if cfg_path.exists():
            try:
                raw = json.loads(cfg_path.read_text(encoding="utf-8"))
                valid_fields = {f for f in cls.__dataclass_fields__ if not f.startswith("_")}
                filtered = {k: v for k, v in raw.items() if k in valid_fields}
                cfg = cls(**filtered, _config_path=cfg_path)
                cfg.ensure_data_dirs()
                return cfg
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        cfg = cls(_config_path=cfg_path)
        cfg.save()
        return cfg
