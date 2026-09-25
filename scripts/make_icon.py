"""Render the app icon (the tray's "effects ON" icon) to an .ico file.

python scripts/make_icon.py build/timbrel.ico
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from timbrel.ui.widgets import state_icon  # noqa: E402


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "build/timbrel.ico")
    out.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])  # noqa: F841 (needed for painting)
    if not state_icon(True, 256).pixmap(256, 256).save(str(out), "ICO"):
        print(f"error: couldn't write {out}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
