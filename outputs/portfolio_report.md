# AI-Driven Global Corporate Portfolio Optimizer

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
| Companies Selected | 25 |
| Countries Represented | 11 |
| Industries Represented | 9 |
| Average AI Strength Score | 0.6294 |
| Total Sales | $4,627.59B |
| Total Profit | $1,275.70B |
| Total Market Value | $34,105.23B |
| Average Profit Margin | 30.63% |

## Top Company Selected

The highest-ranked company in the optimized portfolio is **NVIDIA**, with an AI strength score of **0.8248**.

## Country Exposure

| country        |   count |
|:---------------|--------:|
| United States  |       8 |
| China          |       7 |
| South Korea    |       2 |
| Saudi Arabia   |       1 |
| Taiwan         |       1 |
| Denmark        |       1 |
| Netherlands    |       1 |
| Hong Kong      |       1 |
| Switzerland    |       1 |
| United Kingdom |       1 |
| Germany        |       1 |

## Industry Exposure

| industry                        |   count |
|:--------------------------------|--------:|
| Semiconductors                  |       6 |
| Banking                         |       4 |
| Drugs & Biotechnology           |       4 |
| IT Software & Services          |       3 |
| Technology Hardware & Equipment |       2 |
| Oil & Gas Operations            |       2 |
| Capital Goods                   |       2 |
| Retailing                       |       1 |
| Food, Drink & Tobacco           |       1 |

## Top 10 Companies by AI Strength Score

| company                                  | country       | industry                        |   ai_company_strength_score |
|:-----------------------------------------|:--------------|:--------------------------------|----------------------------:|
| NVIDIA                                   | United States | Semiconductors                  |                    0.824783 |
| Alphabet                                 | United States | IT Software & Services          |                    0.816529 |
| Apple                                    | United States | Technology Hardware & Equipment |                    0.807361 |
| Microsoft                                | United States | IT Software & Services          |                    0.775427 |
| Amazon                                   | United States | Retailing                       |                    0.744365 |
| Saudi Arabian Oil Company (Saudi Aramco) | Saudi Arabia  | Oil & Gas Operations            |                    0.732916 |
| Taiwan Semiconductor                     | Taiwan        | Semiconductors                  |                    0.710999 |
| SK Hynix                                 | South Korea   | Semiconductors                  |                    0.690318 |
| Samsung Electronics                      | South Korea   | Technology Hardware & Equipment |                    0.675191 |
| ICBC                                     | China         | Banking                         |                    0.63692  |

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
