# Third-party notices

Timbrel is MIT licensed (see [LICENSE](LICENSE)). The Windows release bundles the software below, each under its own license. Their license texts are included with each package in the release folder (for example `_internal\*.dist-info\`) and are available at the links.

| Component | License | Link |
| --- | --- | --- |
| Python | PSF License 2.0 | <https://docs.python.org/3/license.html> |
| NumPy | BSD-3-Clause | <https://numpy.org> |
| SciPy | BSD-3-Clause | <https://scipy.org> |
| python-sounddevice | MIT | <https://github.com/spatialaudio/python-sounddevice> |
| PortAudio (bundled with sounddevice) | MIT-style PortAudio license | <https://www.portaudio.com> |
| CFFI | MIT | <https://cffi.readthedocs.io> |
| pycparser | BSD-3-Clause | <https://github.com/eliben/pycparser> |
| Qt 6, via PySide6-Essentials and shiboken6 | LGPL-3.0 | <https://www.qt.io/licensing>, <https://wiki.qt.io/Qt_for_Python> |
| PyInstaller bootloader | GPL-2.0-or-later with the bootloader exception, which allows distributing built programs under any license | <https://pyinstaller.org> |

## Qt / PySide6 (LGPL-3.0)

Timbrel uses Qt and PySide6 under the GNU Lesser General Public License v3. The release is a one-folder build: the Qt and PySide6 libraries are separate, unmodified DLL and `.pyd` files in the `_internal` folder, so you can replace them with your own compatible build. Source code for Qt is available from <https://download.qt.io/> and for PySide6 from <https://code.qt.io/cgit/pyside/pyside-setup.git/>. Timbrel's own source is available in this repository.

## Not included

VB-Audio Virtual Cable is **not** included or redistributed. It is a separate donationware product by VB-Audio Software (<https://vb-audio.com/Cable/>) that users install themselves.
