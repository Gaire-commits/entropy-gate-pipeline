"""Every script must start the way Colab starts it: as a file, from the repo root, with nothing on the path."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py")
                 if p.name != "_common.py" and "argparse" in p.read_text())


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_starts_from_the_repo_root(name):
    """`--help` imports everything the script needs and exits; a missing path setup fails here, not on Colab."""
    out = subprocess.run([sys.executable, f"scripts/{name}", "--help"], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "EGP_DEVICE": "cpu"})
    assert out.returncode == 0, out.stderr[-600:]


def test_a_half_fetched_cache_is_refused(capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    from _common import enough_symbols

    universe = [f"S{i}" for i in range(100)]
    assert enough_symbols({s: 1 for s in universe[:95]}, universe, "data/bars_sip")
    assert not enough_symbols({s: 1 for s in universe[:56]}, universe, "data/bars_sip")
    assert "56 of 100" in capsys.readouterr().out
