import pandas as pd

from app.statement_parser import clean_financial_data


def test_clean_financial_data_normalizes_columns_and_amounts():
    frame = pd.DataFrame(
        {
            " Transaction Date ": ["01/02/2026"],
            "Name": ["Example Store"],
            "Amount": ["19.95"],
        }
    )

    cleaned = clean_financial_data(frame)

    assert list(cleaned.columns) == ["transaction_date", "name", "amount"]
    assert cleaned.loc[0, "amount"] == 19.95


def test_clean_financial_data_handles_empty_input():
    assert clean_financial_data(pd.DataFrame()).empty
