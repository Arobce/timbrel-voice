import os
import subprocess
import sys
import time
import uuid

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from timbrel.ui.single_instance import InstanceServer, notify_running_instance  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def unique_name():
    return f"timbrel-test-{uuid.uuid4().hex}"


def test_no_running_instance(qapp):
    assert not notify_running_instance(unique_name())


def test_second_launch_asks_first_to_show(qapp):
    name = unique_name()
    shown = []
    server = InstanceServer(lambda: shown.append(True), name)
    try:
        assert server.listening
        # A real second process, as when the user launches Timbrel again.
        code = (
            "from PySide6.QtCore import QCoreApplication; QCoreApplication([]); "
            "from timbrel.ui.single_instance import notify_running_instance as n; "
            f"print(n({name!r}))"
        )
        second = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 20
        while second.poll() is None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert second.communicate(timeout=5)[0].strip() == "True"
        qapp.processEvents()
        assert shown == [True]
    finally:
        server.close()


def test_name_is_free_again_after_close(qapp):
    name = unique_name()
    InstanceServer(lambda: None, name).close()
    assert not notify_running_instance(name)
