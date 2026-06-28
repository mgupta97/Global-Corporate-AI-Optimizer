import os
import sys
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from src folder
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.append(SRC_DIR)

from optimizer import build_candidate_pool, optimize_portfolio
from scenarios import SCENARIOS, apply_scenario_score

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "forbes_global_2000_2026_clustered.csv")


st.set_page_config(
    page_title="AI Global Corporate Portfolio Optimizer",
    page_icon="🌍",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def show_metric_cards(portfolio):
    total_sales = portfolio["sales_b"].sum()
    total_profit = portfolio["profit_b"].sum()
    total_market_value = portfolio["market_value_b"].sum()
    avg_profit_margin = portfolio["profit_margin"].mean()
    avg_ai_score = portfolio["ai_company_strength_score"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Companies", f"{len(portfolio)}")
    col2.metric("Countries", f"{portfolio['country'].nunique()}")
    col3.metric("Industries", f"{portfolio['industry'].nunique()}")
    col4.metric("Avg AI Score", f"{avg_ai_score:.3f}")
    col5.metric("Avg Profit Margin", f"{avg_profit_margin:.1%}")

    col6, col7, col8 = st.columns(3)

    col6.metric("Total Sales", f"${total_sales:,.0f}B")
    col7.metric("Total Profit", f"${total_profit:,.0f}B")
    col8.metric("Market Value", f"${total_market_value:,.0f}B")


def plot_exposures(portfolio):
    col1, col2 = st.columns(2)

    with col1:
        country_counts = (
            portfolio["country"]
            .value_counts()
            .reset_index()
        )
        country_counts.columns = ["country", "count"]

        fig_country = px.bar(
            country_counts,
            x="country",
            y="count",
            title="Country Exposure"
        )
        st.plotly_chart(fig_country, use_container_width=True)

    with col2:
        industry_counts = (
            portfolio["industry"]
            .value_counts()
            .reset_index()
        )
        industry_counts.columns = ["industry", "count"]

        fig_industry = px.bar(
            industry_counts,
            x="industry",
            y="count",
            title="Industry Exposure"
        )
        fig_industry.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_industry, use_container_width=True)

    cluster_counts = (
        portfolio["cluster_label"]
        .value_counts()
        .reset_index()
    )
    cluster_counts.columns = ["cluster_label", "count"]

    fig_cluster = px.bar(
        cluster_counts,
        x="cluster_label",
        y="count",
        title="Cluster Exposure"
    )
    fig_cluster.update_layout(xaxis_tickangle=-20)
    st.plotly_chart(fig_cluster, use_container_width=True)


def plot_company_scores(portfolio):
    top_companies = portfolio.sort_values(
        "ai_company_strength_score",
        ascending=False
    ).head(15)

    fig = px.bar(
        top_companies,
        x="company",
        y="ai_company_strength_score",
        color="cluster_label",
        title="Top Companies by AI Strength Score",
        hover_data=[
            "country",
            "industry",
            "rank",
            "sales_b",
            "profit_b",
            "market_value_b",
            "profit_margin"
        ]
    )

    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def show_ai_summary(portfolio):
    top_country = portfolio["country"].value_counts().idxmax()
    top_industry = portfolio["industry"].value_counts().idxmax()
    top_cluster = portfolio["cluster_label"].value_counts().idxmax()

    top_company = portfolio.sort_values(
        "ai_company_strength_score",
        ascending=False
    ).iloc[0]

    st.subheader("AI Analyst Summary")

    st.write(
        f"""
        The optimizer selected **{len(portfolio)} companies** across 
        **{portfolio['country'].nunique()} countries**, 
        **{portfolio['industry'].nunique()} industries**, and 
        **{portfolio['cluster_label'].nunique()} company clusters**.

        The largest country exposure is **{top_country}**, while the most represented 
        industry is **{top_industry}**. The dominant company cluster is 
        **{top_cluster}**.

        The highest-scoring company selected is **{top_company['company']}**, with an 
        AI strength score of **{top_company['ai_company_strength_score']:.3f}**.

        This portfolio balances financial strength, profitability, market value, 
        industry diversification, geographic diversification, and company-type 
        diversification using a constrained optimization model.
        """
    )


def main():
    st.title("🌍 AI-Driven Global Corporate Portfolio Optimizer")
    st.write(
        """
        This dashboard uses Forbes Global 2000 company data to build an optimized 
        global company portfolio using AI-style financial scoring, KMeans clustering, 
        and constrained optimization.
        """
    )

    df = load_data()
    candidate_df = build_candidate_pool(df)

    st.sidebar.header("Optimization Settings")

    scenario_name = st.sidebar.selectbox(
        "Portfolio Strategy",
        list(SCENARIOS.keys())
    )
    st.sidebar.info(SCENARIOS[scenario_name]["description"])


    portfolio_size = st.sidebar.slider(
        "Portfolio Size",
        min_value=10,
        max_value=50,
        value=25,
        step=5
    )

    max_per_country = st.sidebar.slider(
        "Max Companies per Country",
        min_value=2,
        max_value=15,
        value=8
    )

    max_per_industry = st.sidebar.slider(
        "Max Companies per Industry",
        min_value=2,
        max_value=15,
        value=6
    )

    max_per_cluster = st.sidebar.slider(
        "Max Companies per Cluster",
        min_value=2,
        max_value=15,
        value=8
    )

    min_countries = st.sidebar.slider(
        "Minimum Countries",
        min_value=2,
        max_value=15,
        value=7
    )

    min_industries = st.sidebar.slider(
        "Minimum Industries",
        min_value=2,
        max_value=15,
        value=7
    )

    min_clusters = st.sidebar.slider(
        "Minimum Clusters",
        min_value=1,
        max_value=5,
        value=4
    )

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Companies", f"{len(df):,}")
    col2.metric("Eligible Companies", f"{len(candidate_df):,}")
    col3.metric("Countries", f"{df['country'].nunique():,}")
    col4.metric("Industries", f"{df['industry'].nunique():,}")

    if st.button("Run Optimization"):
        try:
            scenario_df = apply_scenario_score(candidate_df, scenario_name)
            portfolio = optimize_portfolio(
                candidate_df=scenario_df,
                portfolio_size=portfolio_size,
                max_per_country=max_per_country,
                max_per_industry=max_per_industry,
                max_per_cluster=max_per_cluster,
                min_countries=min_countries,
                min_industries=min_industries,
                min_clusters=min_clusters,
                objective_col = "scenario_score"
            )

            st.success("Optimization completed successfully.")

            show_metric_cards(portfolio)

            st.subheader("Optimized Portfolio")

            display_cols = [
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
                "scenario_score"
            ]

            st.dataframe(
                portfolio[display_cols].sort_values(
                    "ai_company_strength_score",
                    ascending=False
                ),
                use_container_width=True
            )

            show_ai_summary(portfolio)

            st.subheader("Portfolio Visualizations")
            plot_exposures(portfolio)
            plot_company_scores(portfolio)

            csv = portfolio.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Optimized Portfolio CSV",
                data=csv,
                file_name="optimized_portfolio.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Optimization failed.")
            st.write(str(e))
            st.write(
                """
                Try relaxing the constraints. For example, reduce minimum countries, 
                reduce minimum industries, or increase maximum companies per cluster.
                """
            )


if __name__ == "__main__":
    main()