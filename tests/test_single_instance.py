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


# --- the startup lock (named mutex) ------------------------------------------------

from timbrel.platform.windows import InstanceLock  # noqa: E402


def test_lock_is_exclusive_and_released():
    name = unique_name()
    first = InstanceLock(name)
    assert first.acquired
    second = InstanceLock(name)
    assert not second.acquired
    first.release()
    third = InstanceLock(name)
    assert third.acquired
    third.release()


def test_lock_blocks_another_process():
    name = unique_name()
    lock = InstanceLock(name)
    try:
        code = (
            "from timbrel.platform.windows import InstanceLock; "
            f"print(InstanceLock({name!r}).acquired)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
        )
        assert out.stdout.strip() == "False"
    finally:
        lock.release()


def test_lock_is_freed_when_its_process_exits():
    name = unique_name()
    code = f"from timbrel.platform.windows import InstanceLock; InstanceLock({name!r})"
    subprocess.run([sys.executable, "-c", code], timeout=30, check=True)  # exits without release
    lock = InstanceLock(name)
    assert lock.acquired
    lock.release()


def test_notify_retries_until_the_first_copy_is_listening(qapp):
    name = unique_name()
    shown = []
    code = (
        "from PySide6.QtCore import QCoreApplication; QCoreApplication([]); "
        "from timbrel.ui.single_instance import notify_running_instance as n; "
        f"print(n({name!r}, retry_seconds=10))"
    )
    second = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    # The "first copy" only starts listening a while after the second launch.
    start = time.monotonic()
    while time.monotonic() - start < 1.5:
        qapp.processEvents()
        time.sleep(0.01)
    server = InstanceServer(lambda: shown.append(True), name)
    try:
        deadline = time.monotonic() + 20
        while second.poll() is None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        assert second.communicate(timeout=5)[0].strip() == "True"
        qapp.processEvents()
        assert shown == [True]
    finally:
        server.close()
