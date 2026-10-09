"""Pre-registered configs are frozen: their rules were fixed before any result, so they must never change.

If one of these fails, a pre-registered file was edited. Revert it. A new rule belongs in a new config,
and the hash here is only ever added, never updated.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FROZEN = {
    "ndx_holdout.yaml": "6604d29026d1d1651a1a429861d40a5256486a800c54caa5820ffa4c8b049279",       # 2026-10-07
    "ndx_replication.yaml": "fdfcd479529f999aedf6402ae3044ce4d6599237616a3c29135a8e29efd09fb6",   # 2026-10-07
    "online_sp500.yaml": "7288adbba0f574c883532cc8eacdb43dc79646526e156f1643a9ecd952a19180",      # 2026-10-07
    "e1_volatility.yaml": "0fef2780ed0eecf73afa7584c8488ce3d96b0ad1909422d5fc99c40e80b8da1e",     # 2026-10-08
    "e2_capacity.yaml": "47d2e64292df378a7c992aa4733e3473a3cf2e9fbefe089a6ecc8db854d52bc5",       # 2026-10-08
    "sip_sp500_multihour.yaml": "5436466b6c7d49645d0affdbd5ebe488ee3aee1918a2b5ea4c7d5f66ce538e00",  # 2026-10-09
    "sip_e1_volatility.yaml": "6d5974810d52ab08430a83819740286139e2fcf0c2e05086a773468e9310e4e0",    # 2026-10-09
    "sip_e2_capacity.yaml": "995e294acf4a4c2070f16dc3399d6c4af26f0a79c36a8ae4e822be76138f77e3",      # 2026-10-09
    "sip_ndx_holdout.yaml": "5747b6878724a5b3487557a6294ce6bda00082cc1008907146ac9d8dd29870b3",      # 2026-10-09
    "sip_ndx_replication.yaml": "d6108049b29712a015f64d22537c3feaf571d505627c9be1f98e61e1ea67bd3c",  # 2026-10-09
}


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_pre_registered_config_is_unchanged(name):
    digest = hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest()
    assert digest == FROZEN[name], f"{name} was edited after its rule was fixed; revert it"
