import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


INPUT_PATH = "data/forbes_global_2000_2026_features.csv"
OUTPUT_PATH = "data/forbes_global_2000_2026_scored.csv"


def clip_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def create_ai_strength_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_size_features = [
        "sales_b",
        "profit_b",
        "assets_b",
        "market_value_b",
    ]

    ratio_features = [
        "profit_margin",
        "return_on_assets",
        "asset_turnover",
        "market_to_sales",
    ]

    # Log-transform size features so mega-companies do not dominate too much
    for col in raw_size_features:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Clip extreme ratio outliers
    df = clip_outliers(df, ratio_features)

    scoring_features = [
        "log_sales_b",
        "log_profit_b",
        "log_assets_b",
        "log_market_value_b",
        "profit_margin",
        "return_on_assets",
        "asset_turnover",
        "market_to_sales",
    ]

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df[scoring_features])

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=[f"{col}_score" for col in scoring_features]
    )

    df = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    df["ai_company_strength_score"] = (
        0.18 * df["log_sales_b_score"] +
        0.24 * df["log_profit_b_score"] +
        0.12 * df["log_assets_b_score"] +
        0.24 * df["log_market_value_b_score"] +
        0.10 * df["profit_margin_score"] +
        0.06 * df["return_on_assets_score"] +
        0.03 * df["asset_turnover_score"] +
        0.03 * df["market_to_sales_score"]
    )

    df["ai_rank"] = (
        df["ai_company_strength_score"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    df["forbes_vs_ai_rank_diff"] = df["rank"] - df["ai_rank"]

    return df


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = pd.read_csv(INPUT_PATH)

    df = create_ai_strength_score(df)

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nRobust AI scoring completed successfully!")
    print(f"Saved file: {OUTPUT_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nTop 25 companies by Robust AI Company Strength Score:")
    print(
        df[
            [
                "ai_rank",
                "rank",
                "company",
                "country",
                "industry",
                "sales_b",
                "profit_b",
                "assets_b",
                "market_value_b",
                "profit_margin",
                "return_on_assets",
                "market_to_sales",
                "ai_company_strength_score",
                "forbes_vs_ai_rank_diff",
            ]
        ]
        .sort_values("ai_rank")
        .head(25)
        .to_string(index=False)
    )

    print("\nMost underrated by Robust AI score compared to Forbes rank:")
    print(
        df[
            [
                "rank",
                "ai_rank",
                "company",
                "country",
                "industry",
                "sales_b",
                "profit_b",
                "market_value_b",
                "profit_margin",
                "ai_company_strength_score",
                "forbes_vs_ai_rank_diff",
            ]
        ]
        .sort_values("forbes_vs_ai_rank_diff", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()