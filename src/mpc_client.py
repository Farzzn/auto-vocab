"""MPC-HC 64-bit Web Interface client, Registry helper, and Win32 window tracker."""

from __future__ import annotations

import ctypes
import html
import re
import sys
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

import requests

if sys.platform == "win32":
    import winreg
else:
    winreg = None  # type: ignore[assignment]

MPC_REGISTRY_PATH = r"Software\MPC-HC\MPC-HC\Settings"
MPC_WINDOW_CLASS = "MediaPlayerClassicW"

_VAR_PATTERN = re.compile(
    r'<p\s+[^>]*id=["\'](?P<id>[^"\']+)["\'][^>]*>(?P<val>.*?)</p>',
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class MpcState:
    """Snapshot of MPC-HC playback state from variables.html."""

    connected: bool = False
    state: int = -1  # -1 = none, 0 = stopped, 1 = paused, 2 = playing
    filepath: str = ""
    filename: str = ""
    position_ms: int = 0
    duration_ms: int = 0

    @property
    def is_paused(self) -> bool:
        return self.connected and self.state == 1

    @property
    def is_playing(self) -> bool:
        return self.connected and self.state == 2


def parse_variables_html(html_text: str) -> MpcState:
    """Parse MPC-HC /variables.html HTML body into an MpcState."""
    fields: dict[str, str] = {}
    for match in _VAR_PATTERN.finditer(html_text):
        key = match.group("id").strip().lower()
        val = html.unescape(match.group("val")).strip()
        fields[key] = val

    def _to_int(val: str | None, default: int = 0) -> int:
        if not val:
            return default
        try:
            return int(float(val))
        except ValueError:
            return default

    return MpcState(
        connected=True,
        state=_to_int(fields.get("state"), default=-1),
        filepath=fields.get("filepath", ""),
        filename=fields.get("file", ""),
        position_ms=_to_int(fields.get("position"), default=0),
        duration_ms=_to_int(fields.get("duration"), default=0),
    )


class MpcClient:
    """Polls MPC-HC's built-in Web Interface on localhost."""

    def __init__(self, port: int = 13579, timeout: float = 0.35) -> None:
        self.port = port
        self.timeout = timeout
        self._session = requests.Session()
        self._session.trust_env = False  # bypass system proxies for localhost

    def set_port(self, port: int) -> None:
        self.port = port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/variables.html"

    def poll(self) -> MpcState:
        """Fetch current playback state from MPC-HC."""
        try:
            resp = self._session.get(self.url, timeout=self.timeout)
            if resp.status_code != 200:
                return MpcState(connected=False)
            resp.encoding = resp.encoding or "utf-8"
            return parse_variables_html(resp.text)
        except requests.RequestException:
            return MpcState(connected=False)

    def close(self) -> None:
        self._session.close()


def get_mpc_web_interface_status() -> dict[str, int | bool]:
    """Read MPC-HC Web Interface settings from HKCU Windows Registry."""
    if winreg is None:
        return {"available": False, "enabled": False, "port": 13579, "localhost_only": True}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, MPC_REGISTRY_PATH, 0, winreg.KEY_READ) as key:
            def _read_dword(name: str, default: int) -> int:
                try:
                    val, _ = winreg.QueryValueEx(key, name)
                    return int(val)
                except OSError:
                    return default

            enabled = _read_dword("EnableWebServer", 0) == 1
            port = _read_dword("WebServerPort", 13579)
            localhost_only = _read_dword("WebServerLocalhostOnly", 1) == 1
            return {
                "available": True,
                "enabled": enabled,
                "port": port,
                "localhost_only": localhost_only,
            }
    except OSError:
        return {"available": False, "enabled": False, "port": 13579, "localhost_only": True}


def is_mpc_running() -> bool:
    """Return True if an MPC-HC window is currently open."""
    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.user32.FindWindowW(MPC_WINDOW_CLASS, None))
    except Exception:
        return False


def enable_mpc_web_interface(
    port: int = 13579, allow_while_running: bool = False
) -> tuple[bool, str]:
    """Enable MPC-HC Web Interface (localhost-only) in HKCU Registry."""
    if winreg is None:
        return False, "Windows Registry is only supported on Windows."
    if not allow_while_running and is_mpc_running():
        return (
            False,
            "MPC-HC is currently open. Please close MPC-HC first and click "
            "'Enable MPC-HC Integration' again — otherwise MPC-HC overwrites "
            "this setting when it closes.",
        )
    try:
        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER,
            MPC_REGISTRY_PATH,
            0,
            winreg.KEY_SET_VALUE | winreg.KEY_READ,
        ) as key:
            winreg.SetValueEx(key, "EnableWebServer", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "WebServerLocalhostOnly", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(key, "WebServerPort", 0, winreg.REG_DWORD, int(port))
        return (
            True,
            f"Enabled MPC-HC Web Interface on 127.0.0.1:{port} (localhost-only). "
            "Open MPC-HC to start using the vocabulary assistant.",
        )
    except OSError as exc:
        return False, f"Failed to update MPC-HC registry settings: {exc}"


def _get_window_dpi_scale(hwnd: int) -> float:
    """Return DPI scale factor for a Win32 window handle."""
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "GetDpiForWindow"):
            dpi = user32.GetDpiForWindow(hwnd)
            if dpi > 0:
                return dpi / 96.0
    except Exception:
        pass
    return 1.0


def _get_exact_window_rect_physical(hwnd: int) -> Optional[tuple[int, int, int, int]]:
    """Return physical (left, top, right, bottom) using DWM extended frame if available."""
    rect = wintypes.RECT()
    DWMWA_EXTENDED_FRAME_BOUNDS = 9
    try:
        dwmapi = ctypes.windll.dwmapi
        hr = dwmapi.DwmGetWindowAttribute(
            wintypes.HWND(hwnd),
            wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
            ctypes.byref(rect),
            ctypes.sizeof(rect),
        )
        if hr == 0 and rect.right > rect.left and rect.bottom > rect.top:
            return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        pass

    user32 = ctypes.windll.user32
    if user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        if rect.right > rect.left and rect.bottom > rect.top:
            return (rect.left, rect.top, rect.right, rect.bottom)
    return None


def get_mpc_window_rect(logical: bool = True) -> Optional[tuple[int, int, int, int]]:
    """Find visible, non-minimized MPC-HC window and return (left, top, right, bottom)."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        hwnd = 0
        best_hwnd = 0
        best_area = 0

        # Iterate top-level windows matching MediaPlayerClassicW
        while True:
            hwnd = user32.FindWindowExW(0, hwnd, MPC_WINDOW_CLASS, None)
            if not hwnd:
                break
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                continue
            phys = _get_exact_window_rect_physical(hwnd)
            if phys is None:
                continue
            area = (phys[2] - phys[0]) * (phys[3] - phys[1])
            if area > best_area:
                best_area = area
                best_hwnd = hwnd

        if not best_hwnd:
            # Fallback to FindWindowW
            hwnd = user32.FindWindowW(MPC_WINDOW_CLASS, None)
            if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return None
            best_hwnd = hwnd

        phys = _get_exact_window_rect_physical(best_hwnd)
        if phys is None:
            return None

        if not logical:
            return phys

        scale = _get_window_dpi_scale(best_hwnd)
        return (
            round(phys[0] / scale),
            round(phys[1] / scale),
            round(phys[2] / scale),
            round(phys[3] / scale),
        )
    except Exception:
        return None


def is_mpc_or_own_foreground(own_hwnd: int | None = None) -> bool:
    """Return True if foreground window is MPC-HC or our own overlay window."""
    if sys.platform != "win32":
        return True
    try:
        user32 = ctypes.windll.user32
        fg = user32.GetForegroundWindow()
        if not fg:
            return False
        if own_hwnd and int(fg) == int(own_hwnd):
            return True
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(fg, buf, 256)
        return buf.value == MPC_WINDOW_CLASS
    except Exception:
        return True
