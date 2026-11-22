"""Windows auto-startup registration via HKCU Run Registry key."""

from __future__ import annotations

import sys

from .config import PROJECT_ROOT

if sys.platform == "win32":
    import winreg
else:
    winreg = None  # type: ignore[assignment]

RUN_KEY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME = "MPCVocabAssistant"


def get_startup_command() -> str:
    """Return the silent launcher command registered in Windows Run."""
    vbs_path = PROJECT_ROOT / "run_silent.vbs"
    return f'wscript.exe "{vbs_path}"'


def is_auto_start_enabled() -> bool:
    """Return True if MPCVocabAssistant is configured to start with Windows."""
    if winreg is None:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY_PATH, 0, winreg.KEY_READ) as key:
            val, _ = winreg.QueryValueEx(key, VALUE_NAME)
            return bool(str(val).strip())
    except OSError:
        return False


def set_auto_start(enabled: bool) -> tuple[bool, str]:
    """Enable or disable automatic startup when signing into Windows."""
    if winreg is None:
        return False, "Auto-start is only supported on Windows."
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            RUN_KEY_PATH,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_READ,
        ) as key:
            if enabled:
                cmd = get_startup_command()
                winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, cmd)
                return True, "MPC-HC Vocab Assistant will now start automatically with Windows."
            try:
                winreg.DeleteValue(key, VALUE_NAME)
            except FileNotFoundError:
                pass
            return True, "Removed MPC-HC Vocab Assistant from Windows startup."
    except OSError as exc:
        return False, f"Failed to update Windows startup setting: {exc}"
