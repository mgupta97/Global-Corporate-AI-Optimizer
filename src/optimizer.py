import os
import shutil
import pandas as pd
import pulp


INPUT_PATH = "data/forbes_global_2000_2026_clustered.csv"
OUTPUT_PATH = "outputs/optimized_portfolio.csv"


def build_candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a high-quality candidate pool for optimization.

    Filters remove:
    - Negative-profit companies
    - Very small companies with distorted financial ratios
    - Extreme margin or valuation outliers
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


def get_solver():
    """
    Use system CBC if available.
    Otherwise fall back to PuLP's bundled CBC solver.

    This makes the optimizer more portable across:
    - Local Mac setup
    - GitHub users
    - Streamlit deployment
    - Linux environments
    """

    cbc_path = shutil.which("cbc")

    if cbc_path:
        return pulp.COIN_CMD(path=cbc_path, msg=False)

    return pulp.PULP_CBC_CMD(msg=False)


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

    candidate_df = candidate_df.copy()

    if objective_col not in candidate_df.columns:
        raise ValueError(f"Objective column '{objective_col}' not found in candidate data.")

    n = len(candidate_df)

    model = pulp.LpProblem(
        "AI_Global_Corporate_Portfolio_Optimizer",
        pulp.LpMaximize
    )

    # Decision variable: 1 if company is selected, 0 otherwise
    x = pulp.LpVariable.dicts("select_company", range(n), cat="Binary")

    countries = candidate_df["country"].unique().tolist()
    industries = candidate_df["industry"].unique().tolist()

    has_cluster_column = "company_cluster" in candidate_df.columns
    clusters = candidate_df["company_cluster"].unique().tolist() if has_cluster_column else []

    # Binary variables for represented countries, industries, and clusters
    y_country = pulp.LpVariable.dicts("country_used", countries, cat="Binary")
    y_industry = pulp.LpVariable.dicts("industry_used", industries, cat="Binary")

    if has_cluster_column:
        y_cluster = pulp.LpVariable.dicts("cluster_used", clusters, cat="Binary")

    # Objective: maximize selected score
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
    if has_cluster_column:
        for cluster in clusters:
            idx = candidate_df[candidate_df["company_cluster"] == cluster].index.tolist()

            model += pulp.lpSum(x[i] for i in idx) <= max_per_cluster
            model += pulp.lpSum(x[i] for i in idx) <= len(idx) * y_cluster[cluster]
            model += pulp.lpSum(x[i] for i in idx) >= y_cluster[cluster]

    # Diversification constraints
    model += pulp.lpSum(y_country[c] for c in countries) >= min_countries
    model += pulp.lpSum(y_industry[ind] for ind in industries) >= min_industries

    if has_cluster_column:
        model += pulp.lpSum(y_cluster[c] for c in clusters) >= min_clusters

    solver = get_solver()
    model.solve(solver)

    status = pulp.LpStatus[model.status]
    print(f"\nOptimization status: {status}")

    if status != "Optimal":
        raise RuntimeError(
            "Optimization did not find an optimal solution. "
            "Try relaxing constraints such as minimum countries, minimum industries, "
            "minimum clusters, or maximum companies per country/industry/cluster."
        )

    selected_indices = [i for i in range(n) if x[i].value() == 1]

    optimized_portfolio = candidate_df.loc[selected_indices].copy()
    optimized_portfolio = optimized_portfolio.sort_values(
        objective_col,
        ascending=False
    )

    optimized_portfolio["portfolio_weight"] = 1 / portfolio_size

    return optimized_portfolio


def print_portfolio_summary(portfolio: pd.DataFrame) -> None:
    display_cols = [
        "ai_rank",
        "rank",
        "company",
        "country",
        "industry",
    ]

    if "cluster_label" in portfolio.columns:
        display_cols.append("cluster_label")

    display_cols += [
        "sales_b",
        "profit_b",
        "market_value_b",
        "profit_margin",
        "ai_company_strength_score",
    ]

    if "scenario_score" in portfolio.columns:
        display_cols.append("scenario_score")

    display_cols.append("portfolio_weight")

    print("\nOptimized Portfolio:")
    print(portfolio[display_cols].to_string(index=False))

    print("\nPortfolio Summary:")
    print(f"Companies selected: {len(portfolio)}")
    print(f"Countries represented: {portfolio['country'].nunique()}")
    print(f"Industries represented: {portfolio['industry'].nunique()}")

    if "cluster_label" in portfolio.columns:
        print(f"Clusters represented: {portfolio['cluster_label'].nunique()}")

    print(f"Average AI score: {portfolio['ai_company_strength_score'].mean():.4f}")
    print(f"Total sales: ${portfolio['sales_b'].sum():,.2f}B")
    print(f"Total profit: ${portfolio['profit_b'].sum():,.2f}B")
    print(f"Total market value: ${portfolio['market_value_b'].sum():,.2f}B")
    print(f"Average profit margin: {portfolio['profit_margin'].mean():.2%}")

    print("\nCountry Exposure:")
    print(portfolio["country"].value_counts().to_string())

    print("\nIndustry Exposure:")
    print(portfolio["industry"].value_counts().to_string())

    if "cluster_label" in portfolio.columns:
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