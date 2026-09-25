"""First-run dialog: VB-Cable setup and picking "CABLE Output" in apps (FR12)."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QWidget

from timbrel.platform.windows import VB_CABLE_URL

FIRST_RUN_HTML = f"""
<h2>Welcome to Timbrel</h2>
<p>Timbrel changes your voice in real time and sends it to a <b>virtual
microphone</b> that Discord, games and meeting apps can use.</p>

<h3>1. Install VB-Audio Virtual Cable (free)</h3>
<p>Download it from <a href="{VB_CABLE_URL}">{VB_CABLE_URL}</a>, run the
installer as administrator, then restart your PC. Timbrel sends your voice to
<b>CABLE Input</b>; other apps hear it on <b>CABLE Output</b>.</p>

<h3>2. Pick "CABLE Output" as the microphone in your app</h3>
<ul>
<li><b>Discord:</b> User Settings &rarr; Voice &amp; Video &rarr; Input Device.
Turn off Krisp noise suppression if it muffles the effects.</li>
<li><b>Zoom:</b> Settings &rarr; Audio &rarr; Microphone.</li>
<li><b>Microsoft Teams:</b> Settings &rarr; Devices &rarr; Microphone.</li>
<li><b>Google Meet:</b> Settings &rarr; Audio &rarr; Microphone (allow mic
access in the browser).</li>
</ul>
<p>Meeting apps' own noise suppression can make robot or radio effects sound
choppy; lower it if that happens.</p>

<h3>3. Choose your real microphone here</h3>
<p>Select it under <b>Microphone</b> in Timbrel. The <b>Clean</b> preset keeps
your real voice, just clearer, and is good to leave on for meetings. Press
<b>Bypass</b> any time to go back to your unprocessed voice.</p>

<p><i>Please use voice effects responsibly: don't impersonate others, and check
your workplace's and game's rules.</i></p>
"""


class FirstRunDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Getting started with Timbrel")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        text = QLabel(FIRST_RUN_HTML)
        text.setWordWrap(True)
        text.setOpenExternalLinks(True)
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Got it")
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
