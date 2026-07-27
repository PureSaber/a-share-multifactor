import pandas as pd

from a_share_multifactor.neutralize import neutralize_cross_section


def test_neutralize_by_industry() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01"] * 4),
            "industry": ["银行", "银行", "科技", "科技"],
            "factor": [1.0, 3.0, 2.0, 4.0],
        }
    )
    result = neutralize_cross_section(df, ["factor"], by=["industry"])
    bank_mean = result.loc[result["industry"] == "银行", "factor"].mean()
    tech_mean = result.loc[result["industry"] == "科技", "factor"].mean()
    assert abs(bank_mean) < 1e-9
    assert abs(tech_mean) < 1e-9
