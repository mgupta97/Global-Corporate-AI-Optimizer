import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


SCENARIOS = {
    "Balanced Global Portfolio": {
        "description": "Maximizes overall AI company strength while maintaining country, industry, and cluster diversification."
    },
    "AI & Innovation Portfolio": {
        "description": "Prioritizes semiconductors, software, biotechnology, high-margin innovators, and AI platform leaders."
    },
    "Defensive Quality Portfolio": {
        "description": "Prioritizes profitable, stable, high-margin companies with strong market value and lower dependence on extreme growth assumptions."
    },
    "High-Profit Margin Portfolio": {
        "description": "Prioritizes companies with strong profitability and efficient business models."
    },
    "Non-US Diversified Portfolio": {
        "description": "Reduces dependence on the United States and highlights global leaders from Asia, Europe, and emerging markets."
    }
}


def minmax(series: pd.Series) -> pd.Series:
    values = series.replace([np.inf, -np.inf], np.nan).fillna(series.median())

    scaler = MinMaxScaler()
    return scaler.fit_transform(values.to_frame()).flatten()


def apply_scenario_score(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    df = df.copy()

    df["profit_margin_norm"] = minmax(df["profit_margin"])
    df["return_on_assets_norm"] = minmax(df["return_on_assets"])
    df["market_value_norm"] = minmax(df["market_value_b"])
    df["profit_norm"] = minmax(df["profit_b"])
    df["sales_norm"] = minmax(df["sales_b"])

    innovation_industries = [
        "Semiconductors",
        "IT Software & Services",
        "Drugs & Biotechnology",
        "Technology Hardware & Equipment",
        "Business Services & Supplies"
    ]

    defensive_industries = [
        "Banking",
        "Insurance",
        "Food, Drink & Tobacco",
        "Drugs & Biotechnology",
        "Utilities",
        "Telecommunications Services"
    ]

    df["innovation_bonus"] = df["industry"].isin(innovation_industries).astype(int)

    df["defensive_bonus"] = df["industry"].isin(defensive_industries).astype(int)

    df["non_us_bonus"] = (df["country"] != "United States").astype(int)

    df["high_margin_cluster_bonus"] = (
        df["cluster_label"]
        .isin([
            "High-Margin Innovation Companies",
            "Mega-Cap AI / Platform Leaders"
        ])
        .astype(int)
    )

    if scenario_name == "Balanced Global Portfolio":
        df["scenario_score"] = df["ai_company_strength_score"]

    elif scenario_name == "AI & Innovation Portfolio":
        df["scenario_score"] = (
            0.45 * df["ai_company_strength_score"] +
            0.20 * df["innovation_bonus"] +
            0.15 * df["high_margin_cluster_bonus"] +
            0.10 * df["market_value_norm"] +
            0.10 * df["profit_margin_norm"]
        )

    elif scenario_name == "Defensive Quality Portfolio":
        df["scenario_score"] = (
            0.35 * df["ai_company_strength_score"] +
            0.20 * df["profit_margin_norm"] +
            0.20 * df["profit_norm"] +
            0.15 * df["defensive_bonus"] +
            0.10 * df["market_value_norm"]
        )

    elif scenario_name == "High-Profit Margin Portfolio":
        df["scenario_score"] = (
            0.40 * df["profit_margin_norm"] +
            0.25 * df["return_on_assets_norm"] +
            0.20 * df["ai_company_strength_score"] +
            0.15 * df["profit_norm"]
        )

    elif scenario_name == "Non-US Diversified Portfolio":
        df["scenario_score"] = (
            0.45 * df["ai_company_strength_score"] +
            0.30 * df["non_us_bonus"] +
            0.15 * df["profit_margin_norm"] +
            0.10 * df["market_value_norm"]
        )

    else:
        df["scenario_score"] = df["ai_company_strength_score"]

    return df