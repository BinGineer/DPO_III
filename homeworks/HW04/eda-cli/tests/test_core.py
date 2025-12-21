from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )

def _another_df() -> pd.DataFrame:
    return pd.DataFrame({
    'age': [25, 30, 35, 28, 32, 150, 29, 31, 200, 27],  # 150 и 200 - выбросы
    'salary': [50000, 55000, 60000, 52000, 58000, 1000000, 53000, 56000, 1200000, 51000],  # 1M+ - выбросы
    'score': [75, 80, 85, 78, 82, 95, 77, 81, 200, 76],  # 200 - выброс
    'height': [170, 175, 180, 172, 178, 250, 171, 176, 300, 169],  # 250 и 300 - выбросы
    'department': ['IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'IT', 'HR', 'Sales'],
    'experience': [2, 5, 3, 1, 4, 50, 2, 3, 60, 1],  # 50 и 60 - выбросы
    'sex': [None,None,None,None,'m','f','f',None,None,None],
    'city': ['Moscow','Moscow','Moscow','Moscow','Moscow','Moscow','Moscow','Moscow','Moscow','Moscow']
})
def save():
    _another_df.to_csv('./data/another.csv')

def test_outliners_in():
    df = _another_df()
    summary = summarize_dataset(df)
    miss = missing_table(df)
    qf = compute_quality_flags(summary, miss, 2.5,0.3)
    assert qf["how_many_empties"] == ['sex']
    assert qf["too_many_missing"] == True
    assert qf["may_have_outliers"] == [['score',200.0,'toobig']]
    assert all(df['age'] >= 18) 


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df,2.5,0.3)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

