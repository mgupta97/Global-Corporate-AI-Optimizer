import os
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "outputs/optimized_portfolio.csv"
REPORT_PATH = "outputs/portfolio_report.md"
CHART_DIR = "outputs/charts"


def save_bar_chart(series, title, xlabel, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def create_report(portfolio: pd.DataFrame):
    os.makedirs(CHART_DIR, exist_ok=True)

    country_counts = portfolio["country"].value_counts()
    industry_counts = portfolio["industry"].value_counts()

    top_companies = (
        portfolio.sort_values("ai_company_strength_score", ascending=False)
        .set_index("company")["ai_company_strength_score"]
        .head(10)
    )

    save_bar_chart(
        country_counts,
        "Optimized Portfolio Country Exposure",
        "Country",
        "Number of Companies",
        f"{CHART_DIR}/country_exposure.png"
    )

    save_bar_chart(
        industry_counts,
        "Optimized Portfolio Industry Exposure",
        "Industry",
        "Number of Companies",
        f"{CHART_DIR}/industry_exposure.png"
    )

    save_bar_chart(
        top_companies,
        "Top 10 Companies by AI Strength Score",
        "Company",
        "AI Strength Score",
        f"{CHART_DIR}/top_companies_ai_score.png"
    )

    total_sales = portfolio["sales_b"].sum()
    total_profit = portfolio["profit_b"].sum()
    total_market_value = portfolio["market_value_b"].sum()
    avg_profit_margin = portfolio["profit_margin"].mean()
    avg_ai_score = portfolio["ai_company_strength_score"].mean()

    top_company = portfolio.sort_values(
        "ai_company_strength_score",
        ascending=False
    ).iloc[0]

    report = f"""# AI-Driven Global Corporate Portfolio Optimizer

## Project Summary

This project uses Forbes Global 2000 company data to build an AI-driven optimization system for selecting a globally diversified portfolio of companies.

The system combines:

- Financial feature engineering
- AI-style company strength scoring
- Quality filtering
- Constrained optimization
- Portfolio-level exposure analysis

## Optimized Portfolio Results

| Metric | Value |
|---|---:|
| Companies Selected | {len(portfolio)} |
| Countries Represented | {portfolio["country"].nunique()} |
| Industries Represented | {portfolio["industry"].nunique()} |
| Average AI Strength Score | {avg_ai_score:.4f} |
| Total Sales | ${total_sales:,.2f}B |
| Total Profit | ${total_profit:,.2f}B |
| Total Market Value | ${total_market_value:,.2f}B |
| Average Profit Margin | {avg_profit_margin:.2%} |

## Top Company Selected

The highest-ranked company in the optimized portfolio is **{top_company["company"]}**, with an AI strength score of **{top_company["ai_company_strength_score"]:.4f}**.

## Country Exposure

{country_counts.to_markdown()}

## Industry Exposure

{industry_counts.to_markdown()}

## Top 10 Companies by AI Strength Score

{portfolio.sort_values("ai_company_strength_score", ascending=False)[["company", "country", "industry", "ai_company_strength_score"]].head(10).to_markdown(index=False)}

## Interpretation

The optimizer selected a portfolio dominated by high-quality technology, semiconductor, banking, energy, and consumer companies. The United States and China remain the largest exposures, but diversification constraints forced the model to include additional countries such as South Korea, Japan, Saudi Arabia, Taiwan, Denmark, Hong Kong, and the United Kingdom.

This makes the model more realistic than a simple top-rank selection because it balances company quality with geographic and industry diversification.

## Optimization Logic

The optimization model maximizes total AI company strength score while enforcing the following constraints:

- Select exactly 25 companies
- Limit maximum exposure to any one country
- Limit maximum exposure to any one industry
- Require at least 7 countries
- Require at least 7 industries
- Exclude companies with negative profit
- Exclude very small companies with distorted financial ratios
- Exclude extreme valuation and profitability outliers

## Generated Charts

- `outputs/charts/country_exposure.png`
- `outputs/charts/industry_exposure.png`
- `outputs/charts/top_companies_ai_score.png`
"""

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print("\nPortfolio report created successfully!")
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Charts saved to: {CHART_DIR}")


def main():
    portfolio = pd.read_csv(INPUT_PATH)
    create_report(portfolio)


if __name__ == "__main__":
    main()