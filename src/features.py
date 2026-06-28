import pandas as pd
import numpy as np


DATA_PATH = "data/forbes_global_2000_2026.csv"
OUTPUT_PATH = "data/forbes_global_2000_2026_features.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns={
        "Rank": "rank",
        "Company": "company",
        "Headquarters": "headquarters",
        "Industry": "industry",
        "Sales ($B)": "sales_b",
        "Profit ($B)": "profit_b",
        "Assets ($B)": "assets_b",
        "Market Value ($B)": "market_value_b"
    })

    # Extract actual country from headquarters field
    df["country"] = (
        df["headquarters"]
        .astype(str)
        .str.split(",")
        .str[-1]
        .str.strip()
    )

    return df


def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["industry"] = df["industry"].fillna("Unknown")
    df["market_value_b"] = df["market_value_b"].fillna(df["market_value_b"].median())

    df["profit_margin"] = df["profit_b"] / df["sales_b"]
    df["return_on_assets"] = df["profit_b"] / df["assets_b"]
    df["asset_turnover"] = df["sales_b"] / df["assets_b"]
    df["market_to_sales"] = df["market_value_b"] / df["sales_b"]
    df["market_to_profit"] = df["market_value_b"] / df["profit_b"]

    df = df.replace([np.inf, -np.inf], np.nan)

    ratio_cols = [
        "profit_margin",
        "return_on_assets",
        "asset_turnover",
        "market_to_sales",
        "market_to_profit"
    ]

    for col in ratio_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def main():
    df = pd.read_csv(DATA_PATH)

    df = clean_columns(df)
    df = create_financial_features(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFeature engineering completed successfully!")
    print(f"Saved file: {OUTPUT_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nTop countries:")
    print(df["country"].value_counts().head(15))


if __name__ == "__main__":
    main()