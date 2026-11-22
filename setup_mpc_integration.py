"""CLI helper to enable MPC-HC 64-bit Web Interface in the Windows Registry."""

from __future__ import annotations

from src.config import AppConfig
from src.mpc_client import enable_mpc_web_interface, get_mpc_web_interface_status


def main() -> None:
    cfg = AppConfig.load()
    before = get_mpc_web_interface_status()
    print("Current MPC-HC Web Interface Registry status:", before)

    ok, msg = enable_mpc_web_interface(port=cfg.mpc_port)
    print(msg)
    if ok:
        after = get_mpc_web_interface_status()
        print("Updated status:", after)


if __name__ == "__main__":
    main()
