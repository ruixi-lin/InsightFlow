
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd
from pathlib import Path


def generate_feedback_charts(
    raw_df,
    cleaned_df,
    report,
    output_dir="outputs/charts",
):
    """
    Generate professional Chinese visualizations
    for consumer feedback data.

    Returns
    -------
    chart_paths : dict
        Paths of generated charts.
    """

    if raw_df is None or cleaned_df is None:
        raise ValueError("Input data cannot be None.")

    if raw_df.empty:
        raise ValueError("Raw dataset is empty.")

    if not isinstance(report, dict):
        raise ValueError("Report must be a dictionary.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Unified visual style
    available_fonts = {
        font.name for font in fm.fontManager.ttflist
    }

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Arial Unicode MS",
    ]

    selected_font = next(
        (
            font for font in font_candidates
            if font in available_fonts
        ),
        "DejaVu Sans",
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            selected_font,
            "Microsoft YaHei",
            "SimHei",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#D9E1E8",
        "axes.labelcolor": "#344054",
        "text.color": "#263445",
        "xtick.color": "#526174",
        "ytick.color": "#526174",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.facecolor": "#FFFFFF",
    })

    # Unified color palette
    COLORS = {
        "primary": "#397A9E",
        "secondary": "#69A6A8",
        "accent": "#E49A62",
        "warning": "#D97757",
        "normal": "#69A6A8",
        "text": "#263445",
        "grid": "#E8EDF2",
        "muted": "#9AA8B5",
    }

    chart_paths = {}

    def save_chart(fig, filename):
        """Save chart in high resolution."""
        file_path = output_path / filename

        fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)
        return str(file_path)

    # Chinese anomaly labels
    anomaly_labels = {
        "exact_duplicate": "完全重复记录",
        "duplicate_feedback_id": "重复反馈ID",
        "invalid_rating": "无效评分",
        "invalid_date": "无效日期",
        "missing_feedback_id": "缺失反馈ID",
        "missing_feedback": "缺失反馈文本",
    }

    # 2. Anomaly type distribution
    anomaly_summary = report.get(
        "anomaly_summary", {}
    )

    anomaly_data = pd.Series(
        anomaly_summary,
        dtype="float64",
    )

    anomaly_data = anomaly_data[
        anomaly_data > 0
    ].sort_values()

    fig_height = max(
        4.5,
        0.65 * max(len(anomaly_data), 1) + 1.5,
    )

    fig, ax = plt.subplots(
        figsize=(10, fig_height)
    )

    if not anomaly_data.empty:

        labels = [
            anomaly_labels.get(key, key)
            for key in anomaly_data.index
        ]

        bars = ax.barh(
            labels,
            anomaly_data.values,
            color=COLORS["warning"],
            height=0.55,
            edgecolor="white",
        )

        for bar, value in zip(
            bars, anomaly_data.values
        ):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)} 条",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlim(
            0,
            max(anomaly_data.max() * 1.3, 1.5),
        )

    else:
        ax.text(
            0.5, 0.5,
            "未发现数据异常",
            ha="center",
            va="center",
            fontsize=13,
            transform=ax.transAxes,
        )

    ax.set_title(
        "用户反馈数据异常类型分布",
        pad=18,
    )

    ax.set_xlabel("异常记录数量")
    ax.set_ylabel("异常类型")

    ax.xaxis.grid(
        True,
        linestyle="--",
        alpha=0.5,
        color=COLORS["grid"],
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    chart_paths["anomaly_distribution"] = save_chart(
        fig,
        "anomaly_distribution.png",
    )

    # 3. Data quality distribution
    anomaly_records = report.get(
        "anomaly_records", []
    )

    affected_rows = {
        item.get("source_row")
        for item in anomaly_records
        if item.get("source_row") is not None
    }

    affected_count = min(
        len(affected_rows),
        len(raw_df),
    )

    normal_count = max(
        len(raw_df) - affected_count,
        0,
    )

    total_count = len(raw_df)

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    if total_count > 0:

        values = [
            normal_count,
            affected_count,
        ]

        labels = [
            f"无异常记录（{normal_count}条）",
            f"存在异常记录（{affected_count}条）",
        ]

        ax.pie(
            values,
            labels=labels,
            colors=[
                COLORS["normal"],
                COLORS["warning"],
            ],
            startangle=90,
            counterclock=False,
            autopct=lambda p: (
                f"{p:.1f}%" if p > 0 else ""
            ),
            pctdistance=0.78,
            labeldistance=1.12,
            wedgeprops={
                "width": 0.42,
                "edgecolor": "white",
                "linewidth": 2,
            },
            textprops={
                "fontsize": 10,
            },
        )

        ax.text(
            0,
            0,
            f"{total_count}\n总记录",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=COLORS["text"],
        )

    ax.set_title(
        "用户反馈数据质量分布",
        pad=20,
    )

    ax.axis("equal")

    fig.tight_layout()

    chart_paths["data_quality"] = save_chart(
        fig,
        "data_quality_distribution.png",
    )

    # 4. Rating distribution
    if "rating" in cleaned_df.columns:

        ratings = pd.to_numeric(
            cleaned_df["rating"],
            errors="coerce",
        )

        rating_counts = (
            ratings.dropna()
            .value_counts()
            .reindex(
                [1, 2, 3, 4, 5],
                fill_value=0,
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 5.5)
        )

        bars = ax.bar(
            rating_counts.index.astype(str),
            rating_counts.values,
            color=COLORS["primary"],
            width=0.58,
            edgecolor="white",
        )

        for bar, value in zip(
            bars, rating_counts.values
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.08,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(
            "用户反馈评分分布",
            pad=18,
        )

        ax.set_xlabel("评分等级")
        ax.set_ylabel("反馈记录数量")

        ax.set_ylim(
            0,
            max(rating_counts.max() * 1.3, 1),
        )

        ax.yaxis.grid(
            True,
            linestyle="--",
            alpha=0.5,
            color=COLORS["grid"],
        )

        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(range(5))
        ax.set_xticklabels(
            [
                f"{i}分"
                for i in range(1, 6)
            ]
        )

        fig.tight_layout()

        chart_paths["rating_distribution"] = save_chart(
            fig,
            "rating_distribution.png",
        )

    # 5. Feedback trend over time
    if "date" in cleaned_df.columns:

        dates = pd.to_datetime(
            cleaned_df["date"],
            errors="coerce",
        )

        valid_dates = dates.dropna()

        if not valid_dates.empty:

            daily_dates = valid_dates.dt.normalize()

            full_date_range = pd.date_range(
                start=daily_dates.min(),
                end=daily_dates.max(),
                freq="D",
            )

            trend = (
                daily_dates
                .value_counts()
                .reindex(
                    full_date_range,
                    fill_value=0,
                )
                .sort_index()
            )

            fig, ax = plt.subplots(
                figsize=(10, 5.5)
            )

            ax.plot(
                trend.index,
                trend.values,
                color=COLORS["primary"],
                marker="o",
                markersize=7,
                linewidth=2.5,
            )

            ax.fill_between(
                trend.index,
                trend.values,
                color=COLORS["secondary"],
                alpha=0.12,
            )

            for date, value in trend.items():
                ax.annotate(
                    str(int(value)),
                    xy=(date, value),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_title(
                "用户反馈时间趋势",
                pad=18,
            )

            ax.set_xlabel("日期")
            ax.set_ylabel("每日反馈记录数量")

            if len(trend) <= 14:

                ax.set_xticks(trend.index)

                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter("%m-%d")
                )

            else:

                locator = mdates.AutoDateLocator()

                ax.xaxis.set_major_locator(locator)

                ax.xaxis.set_major_formatter(
                    mdates.ConciseDateFormatter(locator)
                )

            ax.tick_params(
                axis="x",
                rotation=30,
            )

            ax.set_ylim(
                bottom=0,
                top=max(trend.max() * 1.3, 1),
            )

            ax.yaxis.grid(
                True,
                linestyle="--",
                alpha=0.5,
                color=COLORS["grid"],
            )

            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()

            chart_paths["feedback_trend"] = save_chart(
                fig,
                "feedback_trend.png",
            )

    return chart_paths



def generate_sentiment_charts(
    sentiment_df,
    output_dir="outputs/charts",
):
    """
    Generate visualizations for consumer sentiment analysis.
    """

    if sentiment_df is None or sentiment_df.empty:
        raise ValueError("Sentiment dataset is empty.")

    required_columns = {
        "product",
        "date",
        "sentiment_score",
        "sentiment_label",
    }

    missing_columns = required_columns - set(
        sentiment_df.columns
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Configure Chinese font
    available_fonts = {
        font.name for font in fm.fontManager.ttflist
    }

    font_candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans CN",
    ]

    selected_font = next(
        (
            font for font in font_candidates
            if font in available_fonts
        ),
        "DejaVu Sans",
    )

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            selected_font,
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    COLORS = {
        "positive": "#69A6A8",
        "negative": "#D97757",
        "neutral": "#9AA8B5",
        "primary": "#397A9E",
        "grid": "#E8EDF2",
    }

    chart_paths = {}

    sentiment_order = ["正面", "中性", "负面"]

    color_map = {
        "正面": COLORS["positive"],
        "中性": COLORS["neutral"],
        "负面": COLORS["negative"],
    }

    def save_chart(fig, filename):
        file_path = output_path / filename

        fig.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.close(fig)
        return str(file_path)

    # Prepare data
    data = sentiment_df.copy()

    data["sentiment_score"] = pd.to_numeric(
        data["sentiment_score"],
        errors="coerce",
    )

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["sentiment_score", "product"]
    )

    # 2. Overall sentiment distribution
    sentiment_counts = (
        sentiment_df["sentiment_label"]
        .value_counts()
        .reindex(sentiment_order, fill_value=0)
    )

    if "无法判断" in sentiment_df["sentiment_label"].values:
        sentiment_counts["无法判断"] = (
            sentiment_df["sentiment_label"]
            .eq("无法判断")
            .sum()
        )

    fig, ax = plt.subplots(figsize=(8, 7))

    valid_counts = sentiment_counts[
        sentiment_counts > 0
    ]

    if not valid_counts.empty:
        pie_colors = [
            color_map.get(label, "#B0B0B0")
            for label in valid_counts.index
        ]

        ax.pie(
            valid_counts.values,
            labels=valid_counts.index,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={
                "width": 0.42,
                "edgecolor": "white",
                "linewidth": 2,
            },
        )

        ax.text(
            0, 0,
            f"{len(sentiment_df)}\n总反馈",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_title("消费者情感分布", pad=18)
    ax.axis("equal")
    fig.tight_layout()

    chart_paths["sentiment_distribution"] = save_chart(
        fig,
        "sentiment_distribution.png",
    )

    # 3. Sentiment comparison by product
    product_sentiment = pd.crosstab(
        sentiment_df["product"],
        sentiment_df["sentiment_label"],
    )

    product_sentiment = product_sentiment.reindex(
        columns=sentiment_order,
        fill_value=0,
    )

    if "无法判断" in sentiment_df["sentiment_label"].values:
        product_sentiment["无法判断"] = (
            sentiment_df.assign(
                _unjudged=sentiment_df["sentiment_label"]
                .eq("无法判断")
            )
            .groupby("product")["_unjudged"]
            .sum()
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    product_sentiment.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[
            color_map.get(label, "#B0B0B0")
            for label in product_sentiment.columns
        ],
        width=0.62,
        edgecolor="white",
    )

    ax.set_title("不同产品的消费者情感对比", pad=18)
    ax.set_xlabel("产品类型")
    ax.set_ylabel("反馈数量")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="情感类别")

    ax.yaxis.grid(
        True,
        linestyle="--",
        alpha=0.5,
        color=COLORS["grid"],
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    chart_paths["product_sentiment"] = save_chart(
        fig,
        "product_sentiment_comparison.png",
    )

    # 4. Average sentiment score by product
    average_scores = (
        data.groupby("product")["sentiment_score"]
        .mean()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars = ax.barh(
        average_scores.index,
        average_scores.values,
        color=COLORS["primary"],
        height=0.55,
    )

    for bar, value in zip(bars, average_scores.values):
        ax.text(
            value + (0.05 if value >= 0 else -0.05),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    ax.axvline(0, color="#526174", linewidth=0.8)

    ax.set_title("不同产品平均情感得分", pad=18)
    ax.set_xlabel("平均情感得分")
    ax.set_ylabel("产品类型")

    ax.xaxis.grid(
        True,
        linestyle="--",
        alpha=0.5,
        color=COLORS["grid"],
    )

    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    chart_paths["product_average_sentiment"] = save_chart(
        fig,
        "product_average_sentiment.png",
    )

    # 5. Sentiment trend over time
    trend_data = data.dropna(subset=["date"]).copy()

    if not trend_data.empty:

        trend_data["date"] = trend_data["date"].dt.normalize()

        trend = (
            trend_data.groupby("date")["sentiment_score"]
            .mean()
            .sort_index()
        )

        fig, ax = plt.subplots(figsize=(10, 5.5))

        ax.plot(
            trend.index,
            trend.values,
            color=COLORS["primary"],
            marker="o",
            markersize=7,
            linewidth=2.5,
        )

        ax.axhline(
            0,
            color=COLORS["negative"],
            linestyle="--",
            alpha=0.7,
        )

        for date, value in trend.items():
            ax.annotate(
                f"{value:.1f}",
                xy=(date, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )

        ax.set_title("消费者情感变化趋势", pad=18)
        ax.set_xlabel("日期")
        ax.set_ylabel("平均情感得分")

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%m-%d")
        )

        ax.tick_params(axis="x", rotation=30)

        ax.yaxis.grid(
            True,
            linestyle="--",
            alpha=0.5,
            color=COLORS["grid"],
        )

        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        chart_paths["sentiment_trend"] = save_chart(
            fig,
            "sentiment_trend.png",
        )

    return chart_paths
