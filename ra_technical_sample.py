import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


# generate mock data
def generate_mock_datasets(seed: int = 42):
    logger.info("generate data")

    np.random.seed(seed)
    n = 350

    markets = ["market_a", "market_b", "market_c", "market_d"]
    education_levels = ["high_school", "undergraduate", "graduate"]

    df_micro = pd.DataFrame({
        "individual_id": np.arange(1, n + 1),
        "market_tier": np.random.choice(markets, n),
        "education": np.random.choice(education_levels, n, p=[0.4, 0.4, 0.2]),
        "income": np.random.uniform(20, 150, n).round(1),
        "hours_worked": np.random.uniform(10, 60, n).round(0),
        "expenditure": np.random.uniform(5, 80, n).round(2),
        "satisfaction_score": np.random.uniform(1.0, 5.0, n).round(1),
    })

    # noise injection
    df_micro.loc[np.random.choice(df_micro.index, 15, replace=False), "satisfaction_score"] = np.nan
    df_micro.loc[df_micro.index[0], "expenditure"] = 9999.0

    df_wages = pd.DataFrame({
        "worker_id": [101, 102, 103, 104, 105],
        "wage_rate": [25.50, 42.00, 19.75, 31.00, 55.00],
        "firm_id": [1, 2, 1, 3, np.nan],
    })

    df_firms = pd.DataFrame({
        "firm_id": [1, 2, 3, 4],
        "firm_size": ["small", "large", "medium", "large"],
        "industry": ["tech", "finance", "mfg", "tech"],
    })

    df_panel_wide = pd.DataFrame({
        "country": ["Argentina", "Brazil", "Colombia"],
        "gdp_1999": [1200, 3500, 850],
        "gdp_2000": [1250, 3650, 910],
    })

    return df_micro, df_wages, df_firms, df_panel_wide


# clean + transform
def clean_and_transform(df_micro, df_wages, df_firms, df_panel_wide):
    logger.info("clean data")

    df = df_micro.copy()

    # flag outliers
    df["outlier_expenditure"] = df["expenditure"] > 1000

    # clean sample
    df = df[df["satisfaction_score"].notna() & ~df["outlier_expenditure"]]

    # indicators
    df["high_income"] = df["income"] >= 100

    df["income_tier"] = np.select(
        [df["income"] >= 110, df["income"] >= 60],
        ["high", "mid"],
        default="low"
    )

    # subsets
    df_sub_market = df[df["market_tier"].isin(["market_a", "market_b"])]
    df_sub_edu = df[df["education"] == "graduate"]

    # summary stats
    summary_stats = (
        df.groupby("market_tier")
        .agg(
            avg_satisfaction=("satisfaction_score", "mean"),
            std_satisfaction=("satisfaction_score", "std"),
            n=("satisfaction_score", "count"),
            avg_expenditure=("expenditure", "mean"),
        )
        .sort_values("avg_satisfaction", ascending=False)
    )

    # merge example
    df_matched = df_wages.merge(df_firms, on="firm_id", how="left")

    # reshape long
    df_long = pd.melt(
        df_panel_wide,
        id_vars=["country"],
        var_name="year",
        value_name="gdp"
    )

    df_long["year"] = df_long["year"].str.replace("gdp_", "", regex=True).astype(int)

    # reshape wide
    df_wide = df_long.pivot(
        index="country",
        columns="year",
        values="gdp"
    ).reset_index()

    # grouped output
    final_grouped = (
        df.assign(work_intensity=np.where(df["hours_worked"] >= 40, "overtime", "part_time"))
        .groupby("work_intensity")
        .agg(
            n=("satisfaction_score", "count"),
            avg_income=("income", "mean"),
            avg_expenditure=("expenditure", "mean"),
        )
        .reset_index()
    )

    return df, df_long, summary_stats, df_matched, final_grouped


# visualization
def run_visualizations(df):

    logger.info("plot")

    plot_df = df.dropna(subset=["satisfaction_score", "income", "expenditure"]).copy()

    # scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x="income",
        y="expenditure",
        hue="market_tier",
        size="hours_worked",
        alpha=0.7
    )
    plt.title("expenditure vs income")
    plt.tight_layout()
    plt.savefig("scatter_expenditure_income.png", dpi=300)
    plt.close()

    # histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(plot_df["satisfaction_score"], bins=12, kde=True)
    plt.title("satisfaction distribution")
    plt.tight_layout()
    plt.savefig("hist_satisfaction.png", dpi=300)
    plt.close()

    # lmplot
    g = sns.lmplot(
        data=plot_df,
        x="income",
        y="satisfaction_score",
        col="market_tier",
        col_wrap=2,
        height=4
    )

    g.fig.suptitle("satisfaction vs income", y=1.02)
    plt.savefig("lmplot_income_satisfaction.png", dpi=300, bbox_inches="tight")
    plt.close()


# run
if __name__ == "__main__":

    df_micro, df_wages, df_firms, df_panel_wide = generate_mock_datasets()

    df_clean, df_long, summary, merged, final = clean_and_transform(
        df_micro, df_wages, df_firms, df_panel_wide
    )

    run_visualizations(df_clean)

    logger.info("complete")
