import os
import pandas as pd
import pulp


INPUT_PATH = "data/forbes_global_2000_2026_clustered.csv"
OUTPUT_PATH = "outputs/optimized_portfolio.csv"


def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out companies that may distort the optimizer:
    - negative profit companies
    - tiny companies with extreme ratios
    - extreme profit margin outliers
    Also guarantees that country is extracted correctly from headquarters.
    """

    df = df.copy()

    # Safety fix: extract true country if country still contains city + country
    if "headquarters" in df.columns:
        df["country"] = (
            df["headquarters"]
            .astype(str)
            .str.rsplit(",", n=1)
            .str[-1]
            .str.strip()
        )
    else:
        df["country"] = (
            df["country"]
            .astype(str)
            .str.rsplit(",", n=1)
            .str[-1]
            .str.strip()
        )

    candidate_df = df[
        (df["profit_b"] > 0) &
        (df["sales_b"] >= 10) &
        (df["market_value_b"] >= 10) &
        (df["assets_b"] >= 10) &
        (df["profit_margin"] > 0) &
        (df["profit_margin"] <= 0.80) &
        (df["return_on_assets"] <= 0.40) &
        (df["market_to_sales"] <= 30)
    ].copy()

    candidate_df = candidate_df.reset_index(drop=True)

    return candidate_df


def optimize_portfolio(
    candidate_df: pd.DataFrame,
    portfolio_size: int = 25,
    max_per_country: int = 8,
    max_per_industry: int = 6,
    max_per_cluster: int = 8,
    min_countries: int = 7,
    min_industries: int = 7,
    min_clusters: int = 4,
    objective_col: str = "ai_company_strength_score"
) -> pd.DataFrame:

    n = len(candidate_df)

    model = pulp.LpProblem(
        "AI_Global_Corporate_Portfolio_Optimizer",
        pulp.LpMaximize
    )

    # Decision variable: select company or not
    x = pulp.LpVariable.dicts("select_company", range(n), cat="Binary")

    countries = candidate_df["country"].unique().tolist()
    industries = candidate_df["industry"].unique().tolist()
    clusters = candidate_df["company_cluster"].unique().tolist()
    # Binary variables for whether a country/industry is represented
    y_country = pulp.LpVariable.dicts("country_used", countries, cat="Binary")
    y_industry = pulp.LpVariable.dicts("industry_used", industries, cat="Binary")
    y_cluster = pulp.LpVariable.dicts("cluster_used", clusters, cat="Binary")

    # Objective: maximize AI strength score
    model += pulp.lpSum(
        candidate_df.loc[i, objective_col] * x[i]
        for i in range(n)
    )

    # Select exactly N companies
    model += pulp.lpSum(x[i] for i in range(n)) == portfolio_size

    # Country constraints
    for country in countries:
        idx = candidate_df[candidate_df["country"] == country].index.tolist()

        model += pulp.lpSum(x[i] for i in idx) <= max_per_country
        model += pulp.lpSum(x[i] for i in idx) <= len(idx) * y_country[country]
        model += pulp.lpSum(x[i] for i in idx) >= y_country[country]

    # Industry constraints
    for industry in industries:
        idx = candidate_df[candidate_df["industry"] == industry].index.tolist()

        model += pulp.lpSum(x[i] for i in idx) <= max_per_industry
        model += pulp.lpSum(x[i] for i in idx) <= len(idx) * y_industry[industry]
        model += pulp.lpSum(x[i] for i in idx) >= y_industry[industry]

    # Cluster constraints
    for cluster in clusters:
        idx = candidate_df[candidate_df["company_cluster"] == cluster].index.tolist()

        model += pulp.lpSum(x[i] for i in idx) <= max_per_cluster
        model += pulp.lpSum(x[i] for i in idx) <= len(idx) * y_cluster[cluster]
        model += pulp.lpSum(x[i] for i in idx) >= y_cluster[cluster]
    
    # Diversification constraints
    model += pulp.lpSum(y_country[c] for c in countries) >= min_countries
    model += pulp.lpSum(y_industry[ind] for ind in industries) >= min_industries
    model += pulp.lpSum(y_cluster[cluster] for cluster in clusters) >= min_clusters

    solver = pulp.COIN_CMD(
        msg=False,
        path = "/opt/homebrew/bin/cbc"
    )
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    print(f"\nOptimization status: {status}")

    if status != "Optimal":
        raise RuntimeError("Optimization did not find an optimal solution. Try relaxing constraints.")

    selected_indices = [i for i in range(n) if x[i].value() == 1]

    optimized_portfolio = candidate_df.loc[selected_indices].copy()
    optimized_portfolio = optimized_portfolio.sort_values(
        "ai_company_strength_score",
        ascending=False
    )

    optimized_portfolio["portfolio_weight"] = 1 / portfolio_size

    return optimized_portfolio


def print_portfolio_summary(portfolio: pd.DataFrame) -> None:
    print("\nOptimized Portfolio:")
    print(
        portfolio[
            [
                "ai_rank",
                "rank",
                "company",
                "country",
                "industry",
                "cluster_label",
                "sales_b",
                "profit_b",
                "market_value_b",
                "profit_margin",
                "ai_company_strength_score",
                "portfolio_weight"
            ]
        ].to_string(index=False)
    )

    print("\nPortfolio Summary:")
    print(f"Companies selected: {len(portfolio)}")
    print(f"Countries represented: {portfolio['country'].nunique()}")
    print(f"Industries represented: {portfolio['industry'].nunique()}")
    print(f"Average AI score: {portfolio['ai_company_strength_score'].mean():.4f}")
    print(f"Total sales: ${portfolio['sales_b'].sum():,.2f}B")
    print(f"Total profit: ${portfolio['profit_b'].sum():,.2f}B")
    print(f"Total market value: ${portfolio['market_value_b'].sum():,.2f}B")
    print(f"Average profit margin: {portfolio['profit_margin'].mean():.2%}")

    print("\nCountry Exposure:")
    print(portfolio["country"].value_counts().to_string())

    print("\nIndustry Exposure:")
    print(portfolio["industry"].value_counts().to_string())

    print("\nCluster Exposure:")
    print(portfolio["cluster_label"].value_counts().to_string())

def main():
    df = pd.read_csv(INPUT_PATH)

    candidate_df = build_candidate_pool(df)

    print("\nCandidate pool created.")
    print(f"Original companies: {len(df)}")
    print(f"Eligible companies after quality filters: {len(candidate_df)}")

    portfolio = optimize_portfolio(candidate_df)

    os.makedirs("outputs", exist_ok=True)
    portfolio.to_csv(OUTPUT_PATH, index=False)

    print_portfolio_summary(portfolio)

    print(f"\nSaved optimized portfolio to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()