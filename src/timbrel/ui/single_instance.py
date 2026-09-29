"""One Timbrel at a time: a second launch brings up the running window.

Two copies would fight over the (exclusively opened) devices and both react
to the global hotkeys, so the first copy listens on a per-user local socket;
later launches send it "show", wait for its "ok", and exit.
"""

from __future__ import annotations

import getpass
import time
from collections.abc import Callable
from functools import partial

from PySide6.QtCore import QObject
from PySide6.QtNetwork import QLocalServer, QLocalSocket

CONNECT_TIMEOUT_MS = 500
ACK_TIMEOUT_MS = 1000


def server_name() -> str:
    return f"timbrel-{getpass.getuser()}"


def notify_running_instance(name: str | None = None, retry_seconds: float = 0.0) -> bool:
    """True if another copy is running (and was asked to show itself).

    Waits for the running copy to acknowledge: closing the socket straight
    after writing can drop the message before it's read. With
    ``retry_seconds``, keeps trying while that copy is still starting up.
    """
    deadline = time.monotonic() + retry_seconds
    while True:
        socket = QLocalSocket()
        socket.connectToServer(name or server_name())
        if socket.waitForConnected(CONNECT_TIMEOUT_MS):
            socket.write(b"show\n")
            socket.waitForBytesWritten(CONNECT_TIMEOUT_MS)  # on Windows the write needs this
            socket.waitForReadyRead(ACK_TIMEOUT_MS)
            socket.abort()
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.2)


class InstanceServer(QObject):
    """Listens for later launches and calls ``on_show`` when one arrives."""

    def __init__(self, on_show: Callable[[], None], name: str | None = None) -> None:
        super().__init__()
        self._on_show = on_show
        self._server = QLocalServer(self)
        self._server.newConnection.connect(self._accept)
        name = name or server_name()
        if not self._server.listen(name):
            # A crashed copy can leave the name behind; nobody answered, so reclaim it.
            QLocalServer.removeServer(name)
            self._server.listen(name)

    @property
    def listening(self) -> bool:
        return self._server.isListening()

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            # The socket lives until its message is read; it's a child of the
            # server, so any leftovers go when the server closes.
            socket.readyRead.connect(partial(self._read, socket))
            if socket.bytesAvailable():  # the message may already be here
                self._read(socket)

    def _read(self, socket: QLocalSocket) -> None:
        if b"show" in bytes(socket.readAll()):
            socket.write(b"ok\n")
            socket.flush()
            socket.disconnectFromServer()
            socket.deleteLater()
            self._on_show()

    def close(self) -> None:
        self._server.close()
