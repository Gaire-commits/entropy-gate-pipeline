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
    "sip_e5_leadlag.yaml": "1c21cc07bd4e0994bbd72768793f57813fd29b6e2c93b7e72ef04788a5c86cff",        # 2026-10-13
    "sip_e6_reversal.yaml": "41f7f1c3fd017e47c66c2d70ebfb6523eb0fa447c99c92fc809cbf7f17bd0820",       # 2026-10-13
    "sip_e7_slope.yaml": "fa6011f5603c1b992a3dc5aad519fcd3839b470ec8f51b0839d600ccbcea568f",          # 2026-10-14
    "sip_e8_scaled_slope.yaml": "d8dd543f53e34235ea03521104aaddaec4bde888ff3b1a9c1747568e9e159433",   # 2026-10-14
    "sip_e9_horizons.yaml": "d677a329be13e31c92095369474cfbff64492b8460ab3e6b211cb7f637ae39ae",       # 2026-10-14
    "sip_e10_holds.yaml": "dbe879df1638da7bb51d0d9e220ebedffda527cfd6b4020198448e22897dcc0a",         # 2026-10-14
    "sip_e11_midcap.yaml": "232401ca0a48826a29216213686170962594ff487902b306ab8a9af79b175576",        # 2026-10-14
    "sip_e12_peers.yaml": "babb692d8230829171f3cb530bd96a1c723813896f68dd6c3abe2d036fb77a64",         # 2026-10-14
    "sip_pe_free.yaml": "cd1b4a9825394cb76ede241e257203b293f615205ead63aa0924f604b637a993",             # 2026-10-25
}


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_pre_registered_config_is_unchanged(name):
    digest = hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest()
    assert digest == FROZEN[name], f"{name} was edited after its rule was fixed; revert it"


# The first freeze of the two protocols sharing the reserved period (2026-11-02). Both are superseded by version 2 (protocols/amendment_1.md)
# before that period was scored. These files stay as they were; the code they ran on is at commit 8ee50cb.
SUPERSEDED = {
    "protocols/portfolio_v1.yaml": "40d8c452ac6f0e72",
    "protocols/volatility_v1.yaml": "709be89fabba145a",
}


@pytest.mark.parametrize("path", sorted(SUPERSEDED))
def test_superseded_protocols_are_unchanged(path):
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest().startswith(SUPERSEDED[path])


# Version 2 of both protocols, its amendment and verification, and the code that runs them. A failure means one of these changed after the
# freeze: revert it, or file a new dated amendment with its reason before anything is scored.
PROTOCOLS = {
    "protocols/portfolio_v2.yaml": "9fde1f833b1c26ab5b03e3586b078122678c09be5021f82a27a7e9bd0d4bbae8",
    "protocols/volatility_v2.yaml": "aba93c82953dc1b230087f052a4694fccd8929e421921b1b7feb387b15526cfd",
    "protocols/amendment_1.md": "8d00546bda08a517db2d6a1fdbcd15a322457fca1cb78c032440aab1f90a1ad3",
    "protocols/clark_west_verification.md": "8107e1a038d27ac69cbc64d2cbc4654871f4c5f312340e70265cf6954a745a8b",
    "src/portfolio.py": "8d7663681a576aacc3e0f88c57b8302113d1c5dd16fe7250913b807ecd0e62b7",
    "src/volforecast.py": "0d818bd704df1409dec241866e7427d942e6abf5867b05c00462b5e78db6c02d",
    "src/volpanel.py": "a485af9dd7160b12a08bf3e2a0cb5d645ea4ebef58b942841206fc0179d79066",
    "src/official.py": "097fbf2c494314902255b2b779d9228a3cdff87f73bab1d39c2a78fd2bae480a",
    "src/nyse.py": "083ec64b7e0902fa8c4c46a4f9c0c22f4ef76d5a4e54b01c5de0bdc6cbca0f15",
    "src/audit.py": "e64b6ebbb8d1e4787dd11c4d7fbd246a3f0a797970b1516acfc3c075571e8d7d",
    "scripts/portfolio_rules.py": "5f6e8c2754dc681a87d691d1fda715ec1bd1899d8cf75e16a84427c3da45c6f8",
    "scripts/reserved_portfolio.py": "3aeec125d92d398272b29fa8925abe6755dae9de94915aa2be53fa4e7f6d1034",
    "scripts/vol_forecast.py": "dd6f8178253dc030fbd5ec635811a62fbd659500df6f866bfd3f8edd462c888d",
    "scripts/protocol_checks.py": "b5968116672b6e2163d7eb750267507f035691770ee64fba0a8253465a18f6f6",
    "scripts/official_prices.py": "497cb3f734b4b3ada75849da4d4cd4766ca627b9026c6516b5585d98e9fc6cde",
    "scripts/build_sessions.py": "94d83da6ee05f6f592509762731d0d27828e875b36e1a92ba990bdc920de0087",
    "scripts/cw_verification.py": "a192948c5ccbc10189cf9756071f75a8baeb88e5ace837890b89eff922c88687",
    "registry/vol_registry.csv": "3d082a611182bb5d26282b0825c51483327fa94180034cdfd42fbf9f62e2cf7f",
    "registry/sp500_join_dates.csv": "b6a6c3bd707ba91973ddbbc3fa0e36040442301c6c030b3d06212aee77eaed40",
    "universe/sp500.csv": "fdbc103d4f552caf475f42411eccf0c4712f96089650d5d565763e174b6040c4",
}


@pytest.mark.parametrize("path", sorted(PROTOCOLS))
def test_frozen_protocol_and_its_code_are_unchanged(path):
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert digest == PROTOCOLS[path], f"{path} changed after the protocols were frozen; revert it or file a new dated amendment"


def test_the_protocols_match_the_code_that_runs_them():
    import sys

    import yaml
    sys.path.insert(0, str(ROOT))
    from src import portfolio as pf, volforecast as vf
    cfg, doc = pf.load_protocol(ROOT / "protocols" / "portfolio_v2.yaml")
    assert pf.step(cfg) == 7 and (cfg.rebalance_every, cfg.horizon, cfg.tranches, cfg.top_frac, cfg.hold_frac) == (21, 21, 3, 0.20, 0.40)
    assert (cfg.holdout_start, cfg.holdout_end) == ("2024-01-03", "2026-10-12")
    assert (cfg.hac_lags, cfg.cw_critical, cfg.cw_critical_holm) == (2, 1.645, 1.96)
    assert set(doc["benchmark_reference"]) == {"SPY", "BIL"}
    with open(ROOT / "protocols" / "volatility_v2.yaml") as f:
        vol = yaml.safe_load(f)
    default = vf.VolConfig()
    assert vf.VolConfig(**vol["settings"]) == default
    assert vol["baseline"]["form"] in vf.CANDIDATES
    assert (default.hac_lags, default.cw_critical, default.cw_critical_holm) == (21, 1.94, 2.33)
    assert (default.test_start, default.test_end) == (cfg.holdout_start, cfg.holdout_end)
    assert (default.pe_m, default.pe_tau, default.pe_window) == (cfg.pe_m, cfg.pe_tau, cfg.pe_window)
    assert default.validation_start == cfg.validation_start
    for name in ("portfolio_v2.yaml", "volatility_v2.yaml", "amendment_1.md", "clark_west_verification.md"):
        text = (ROOT / "protocols" / name).read_text()
        assert "@@" not in text, f"{name} still holds a placeholder"
        assert name.endswith(".md") or "reserved for this specification but previously exposed through related research" in text.lower()
