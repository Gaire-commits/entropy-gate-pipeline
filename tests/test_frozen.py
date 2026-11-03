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


# The two protocols sharing the reserved period, and everything that decides their results (frozen 2026-11-02, before it was scored).
# A change to any of these means a new dated protocol with the reason in it, and a new entry here; never an updated hash.
PROTOCOLS = {
    "protocols/portfolio_v1.yaml": "40d8c452ac6f0e72b116155cd7ead9e67c65041cf34c555830c94163ec73b846",
    "protocols/volatility_v1.yaml": "709be89fabba145a74e78c90ffb1d02e5d9813eb5211ac6e3f24b5f77be1fd7e",
    "src/portfolio.py": "95775b6e4f94a84678151d7467bf6e9aa001629ef4bde818cfb1534a0811225d",
    "src/volforecast.py": "2633579b09b665e8d6f9fbd84aa19b03f39e9920b3539f64926e8562f52405f9",
    "src/volpanel.py": "5c8c477e138ee82362af25862ed559ebed1b276d53848b666b35373d4b838525",
    "src/official.py": "18ef182ed096f8b228ffeb314d64849caba6a04566d6d09b2bd84acbbcc096f3",
    "src/nyse.py": "083ec64b7e0902fa8c4c46a4f9c0c22f4ef76d5a4e54b01c5de0bdc6cbca0f15",
    "scripts/portfolio_rules.py": "5f6e8c2754dc681a87d691d1fda715ec1bd1899d8cf75e16a84427c3da45c6f8",
    "scripts/reserved_portfolio.py": "63d54b41852b375f32875f3889b8b669f0911051ce7c09ac3d32d2e3bc4d69bf",
    "scripts/vol_forecast.py": "1024e7c44c11d68e688a690a80ff2646e9e1488a73e65877fc02d89060279d9e",
    "scripts/protocol_checks.py": "42d6b20b2af082c84b9bd9da6e62e67bcc953d603270eb2c92b5fcb6266ba4a4",
    "scripts/official_prices.py": "ed14df5f28037cbae54c2f8a9d3d237cf208f53eed6efa61a5f1448060f7578b",
    "registry/vol_registry.csv": "3d082a611182bb5d26282b0825c51483327fa94180034cdfd42fbf9f62e2cf7f",
    "registry/sp500_join_dates.csv": "b6a6c3bd707ba91973ddbbc3fa0e36040442301c6c030b3d06212aee77eaed40",
    "universe/sp500.csv": "fdbc103d4f552caf475f42411eccf0c4712f96089650d5d565763e174b6040c4",
}


@pytest.mark.parametrize("path", sorted(PROTOCOLS))
def test_frozen_protocol_and_its_code_are_unchanged(path):
    digest = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    assert digest == PROTOCOLS[path], f"{path} changed after the protocols were frozen; revert it or file a new dated protocol"


def test_the_protocols_match_the_code_that_runs_them():
    import sys

    import yaml
    sys.path.insert(0, str(ROOT))
    from src import portfolio as pf, volforecast as vf
    cfg, doc = pf.load_protocol(ROOT / "protocols" / "portfolio_v1.yaml")
    assert pf.step(cfg) == 7 and (cfg.rebalance_every, cfg.horizon, cfg.tranches, cfg.top_frac, cfg.hold_frac) == (21, 21, 3, 0.20, 0.40)
    assert (cfg.holdout_start, cfg.holdout_end) == ("2024-01-03", "2026-10-12")
    assert set(doc["benchmark_reference"]) == {"SPY", "BIL"}
    with open(ROOT / "protocols" / "volatility_v1.yaml") as f:
        vol = yaml.safe_load(f)
    assert vf.VolConfig(**vol["settings"]) == vf.VolConfig()
    assert vol["baseline"]["form"] in vf.CANDIDATES
    assert (vf.VolConfig().test_start, vf.VolConfig().test_end) == (cfg.holdout_start, cfg.holdout_end)
    assert (vf.VolConfig().pe_m, vf.VolConfig().pe_tau, vf.VolConfig().pe_window) == (cfg.pe_m, cfg.pe_tau, cfg.pe_window)
