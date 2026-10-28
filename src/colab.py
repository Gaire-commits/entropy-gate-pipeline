"""Colab helpers: push results to Drive, check Drive is really there, and keep a copy that does not depend on it.

Colab writes to Drive through a mount that uploads in the background. In this project's runtimes the upload
has been failing quietly: since about 2026-10-14 Drive shows the folders runs create but none of the files
(HEAT's registered run printed "could not flush Drive" and "Mountpoint must not already contain files").
So every result is also zipped and downloaded to your computer, the flush is time-limited so it never
hangs a session, and `drive_status` reports the mount and the space Drive says is left (a full Drive
accepts folders but refuses files).
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import zipfile

MOUNT = "/content/drive"


def in_colab() -> bool:
    return "google.colab" in sys.modules


def drive_status(mount: str = MOUNT) -> str:
    if not os.path.exists(mount):
        return f"{mount} does not exist: Drive is not mounted (run section 1)."
    if not os.path.ismount(mount):
        return (f"{mount} is a plain folder, NOT a Drive mount: anything written there stays on this machine. "
                "Restart the runtime (Runtime -> Restart session) and run section 1 again.")
    total, used, free = shutil.disk_usage(mount)
    note = " Drive reports almost no free space: it will accept folders but refuse files." if free < 1e9 else ""
    return f"Drive mounted at {mount}: {used / 1e9:.1f} GB used of {total / 1e9:.1f} GB, {free / 1e9:.1f} GB free.{note}"


def sync_drive(timeout: int = 300, mount: str = MOUNT) -> str:
    """Flush pending uploads and remount. Never hangs: after `timeout` seconds it returns and the upload
    carries on in the background while the session lives."""
    if not in_colab():
        return "not in Colab: nothing to sync"
    from google.colab import drive

    failed: list[Exception] = []

    def flush():
        try:
            drive.flush_and_unmount()
        except Exception as exc:  # noqa: BLE001
            failed.append(exc)

    t = threading.Thread(target=flush, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return (f"Drive still uploading after {timeout}s; it keeps going while this session lives. "
                "Your downloaded zip is the copy to rely on.")
    try:
        drive.mount(mount)
    except Exception as exc:  # noqa: BLE001
        return f"flushed, but could not remount ({exc}); run drive.mount('{mount}') yourself"
    return f"could not flush Drive: {failed[0]}" if failed else "Drive flushed and remounted."


def download_results(path: str, name: str, keep: tuple = (".md", ".csv", ".json", ".parquet"),
                     max_file_mb: float = 150, max_total_mb: float = 600) -> str | None:
    """Zip the result files under `path` and download the zip to your computer (it lands in Downloads)."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"nothing to download: {path} does not exist on this runtime")
        return None
    target = f"/content/{name}.zip" if in_colab() else os.path.join(os.path.dirname(path), f"{name}.zip")
    total, skipped, n = 0.0, [], 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(path):
            for f in sorted(files):
                full = os.path.join(root, f)
                size = os.path.getsize(full) / 1e6
                if not f.endswith(keep):
                    continue
                if size > max_file_mb or total + size > max_total_mb:
                    skipped.append(f"{os.path.relpath(full, path)} ({size:.0f} MB)")
                    continue
                z.write(full, os.path.relpath(full, path))
                total, n = total + size, n + 1
    print(f"{n} files, {total:.1f} MB before compression, in {target}")
    if skipped:
        print("left out (too large; still on this runtime and Drive):", ", ".join(skipped))
    if in_colab():
        from google.colab import files
        files.download(target)
    return target
