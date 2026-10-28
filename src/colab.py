"""Colab helpers: check Drive is really there, keep a copy of results that does not depend on it, and push to Drive
at the end of a session.

Colab writes to Drive through a mount that uploads in the background. In this project's runtimes the upload
has been failing quietly: since about 2026-10-14 Drive shows the folders runs create but none of the files
(HEAT's registered run printed "could not flush Drive" and "Mountpoint must not already contain files").
So every result is zipped and downloaded to your computer.

Flushing to Drive means unmounting it, and while Drive is unmounted `data/` and `outputs/` (links into it)
point at nothing. So `sync_drive` is only for the end of a session, never between sections, and
`ensure_drive` remounts Drive before a section that needs it.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import zipfile

MOUNT = "/content/drive"
LINKS = ("data", "outputs")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def drive_status(mount: str = MOUNT) -> str:
    if not os.path.exists(mount):
        return f"{mount} does not exist: Drive is not mounted (run section 1)."
    if not os.path.ismount(mount):
        return (f"{mount} is NOT a Drive mount: run section 1. If mounting fails with 'Mountpoint must not already "
                "contain files', restart the session (Runtime -> Restart session) and run sections 1-3.")
    total, used, free = shutil.disk_usage(mount)
    note = " Drive reports almost no free space: it will accept folders but refuse files." if free < 1e9 else ""
    return f"Drive mounted at {mount}: {used / 1e9:.1f} GB used of {total / 1e9:.1f} GB, {free / 1e9:.1f} GB free.{note}"


def ensure_drive(mount: str = MOUNT) -> bool:
    """Mount Drive if it is not mounted, then check that data/ and outputs/ reach it."""
    if in_colab() and not os.path.ismount(mount):
        from google.colab import drive
        drive.mount(mount)
    broken = [name for name in LINKS if os.path.islink(name) and not os.path.exists(name)]
    print(drive_status(mount))
    if broken:
        print(f"STOP: {', '.join(broken)} point at Drive but Drive is not reachable. Run sections 1-3 again.")
        return False
    return True


def sync_drive(timeout: int = 900, mount: str = MOUNT) -> str:
    """End of session only: flush pending uploads (this unmounts Drive), then mount it again."""
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
        return (f"Drive is still flushing after {timeout}s and will unmount when it finishes. Before running any "
                "other cell, run section 1 to mount it again. The downloaded zips are the copies to rely on.")
    try:
        drive.mount(mount)
    except Exception as exc:  # noqa: BLE001
        return f"flushed, but Drive could not be mounted again ({exc}): restart the session and run sections 1-3"
    return f"could not flush Drive: {failed[0]}" if failed else "Drive flushed and mounted again."


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
    if n == 0:
        print(f"WARNING: {path} has no result files on this runtime, so the zip is empty (they were never written "
              "here, or Drive lost them). Rerun the section that writes them.")
    if skipped:
        print("left out (too large; still on this runtime):", ", ".join(skipped))
    if in_colab() and n:
        from google.colab import files
        files.download(target)
    return target
