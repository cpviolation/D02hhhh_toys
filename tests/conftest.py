import os

# several test processes may share one GPU: do not let JAX preallocate its memory
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest

from d02hhhh_toys.model import AmplitudeModel

PHI_RHO_S = "D0{rhoOmega00{rho(770)0[GounarisSakurai]{pi+,pi-}},phi(1020)0{K+,K-}}"
KST_KST_P = "D0[P]{K*(892)0{K+,pi-},K*(892)bar0{K-,pi+}}"
PHI_PIPI_S = "D0{phi(1020)0{K+,K-},PiPi20[kMatrix.pole.0]{pi+,pi-}}"
K1680_KPI4 = "D0{K*(1680)0{K+,pi-},KPi40[FOCUS.Kpi]{K-,pi+}}"
K1680BAR_KPI5 = "D0{K*(1680)bar0{K-,pi+},KPi50[FOCUS.Kpi]{K+,pi-}}"
SUBSET = [PHI_RHO_S, KST_KST_P, PHI_PIPI_S, K1680_KPI4, K1680BAR_KPI5]


@pytest.fixture(scope="session")
def subset_model():
    """Five chains with their default couplings: CP-even, CP-odd and a CP-mirror pair."""
    return AmplitudeModel(chains=SUBSET)


#: backend of the full model: D02HHHH_TEST_BACKEND=jax uses the GPU (run without -n)
BACKEND = os.environ.get("D02HHHH_TEST_BACKEND", "numpy")


@pytest.fixture(scope="session")
def full_model():
    return AmplitudeModel(backend=BACKEND)
