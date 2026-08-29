from importlib.metadata import PackageNotFoundError, distribution, version

import akshare
from py_mini_racer import MiniRacer


def _namespace_owners() -> list[str]:
    owners = []
    for name in ("mini-racer", "py-mini-racer", "akracer"):
        try:
            files = distribution(name).files or ()
        except PackageNotFoundError:
            continue
        if any(str(path).replace("\\", "/").startswith("py_mini_racer/") for path in files):
            owners.append(name)
    return owners


def test_akshare_uses_one_working_mini_racer_provider() -> None:
    # The derivative is metadata-only: AKShare's runtime payload remains official 1.18.88.
    assert akshare.__version__ == "1.18.88"
    assert version("akshare") == "1.18.88.post1"
    assert _namespace_owners() == ["mini-racer"]
    assert MiniRacer().eval("1+1") == 2
