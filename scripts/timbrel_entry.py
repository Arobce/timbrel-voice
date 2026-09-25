"""PyInstaller entry point for the Windows build (see scripts/build.ps1)."""

import sys

from timbrel.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
