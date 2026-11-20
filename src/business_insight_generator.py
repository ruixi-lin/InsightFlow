
from pathlib import Path
from datetime import datetime
import pandas as pd


def _split_items(value):
    """Split multi-label text into unique items."""
    if pd.isna(value):
        return []

    items = (
        str(value)
        .replace("，", "、")
        .replace(",", "、")
        .split("、")
    )

    return list(dict.fromkeys(
        item.strip() for item in items if item.strip()
    ))


def _safe_cell(value):
    """Prepare text for Markdown tables."""
    if pd.isna(value):
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(headers, rows):
    """Build a Markdown table without extra dependencies."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    for row in rows:
        lines.append(
            "| " + " | ".join(_safe_cell(v) for v in row) + " |"
        )

    return lines


def generate_business_insights(
    feedback_df,
    sentiment_df,
    classification_df,
    output_dir="outputs/reports",
    filename="business_insight_report.md",
):
    """
    Generate a product-oriented business insight report.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Validate and merge analysis results
    if feedback_df is None or feedback_df.empty:
        raise ValueError("Feedback dataset is empty.")

    if "feedback_id" not in feedback_df.columns:
        raise ValueError("Missing feedback_id in feedback data.")

    data = feedback_df.copy()

    if data["feedback_id"].duplicated().any():
        raise ValueError("Duplicate feedback_id in raw feedback data.")

    for result_df in [sentiment_df, classification_df]:

        if result_df is None or result_df.empty:
            raise ValueError("Analysis result is empty.")

        if "feedback_id" not in result_df.columns:
            raise ValueError(
                "Analysis results must contain feedback_id."
            )

        if result_df["feedback_id"].duplicated().any():
            raise ValueError(
                "Duplicate feedback_id in analysis results."
            )

        extra_columns = [
            col for col in result_df.columns
            if col != "feedback_id" and col not in data.columns
        ]

        data = data.merge(
            result_df[["feedback_id"] + extra_columns],
            on="feedback_id",
            how="left",
            validate="one_to_one"
        )

    total = len(data)

    # 2. Overall sentiment analysis
    if "sentiment_label" in data.columns:
        sentiment_counts = (
            data["sentiment_label"]
            .fillna("无法判断")
            .value_counts()
            .to_dict()
        )
    else:
        sentiment_counts = {}

    positive = sentiment_counts.get("正面", 0)
    negative = sentiment_counts.get("负面", 0)
    neutral = sentiment_counts.get("中性", 0)
    unknown = total - positive - negative - neutral

    # 3. Product performance analysis
    data["product"] = data.get(
        "product",
        pd.Series("未注明产品", index=data.index)
    ).fillna("未注明产品")

    if "rating" in data.columns:
        data["rating"] = pd.to_numeric(
            data["rating"], errors="coerce"
        )
    else:
        data["rating"] = float("nan")

    if "sentiment_score" in data.columns:
        data["sentiment_score"] = pd.to_numeric(
            data["sentiment_score"], errors="coerce"
        )
    else:
        data["sentiment_score"] = float("nan")

    product_summary = data.groupby("product").agg(
        feedback_count=("feedback_id", "count"),
        average_rating=("rating", "mean"),
        average_sentiment=("sentiment_score", "mean")
    ).round(2)

    # 4. Category analysis
    category_counts = {}

    if "categories" in data.columns:
        for value in data["categories"]:
            for category in _split_items(value):
                category_counts[category] = (
                    category_counts.get(category, 0) + 1
                )

    # 5. Pain-point analysis by product
    pain_point_counts = {}
    product_pain_points = {}

    if "pain_points" in data.columns:
        for _, row in data.iterrows():

            product = row["product"]
            points = _split_items(row["pain_points"])

            for point in points:
                pain_point_counts[point] = (
                    pain_point_counts.get(point, 0) + 1
                )

                if product not in product_pain_points:
                    product_pain_points[product] = {}

                product_pain_points[product][point] = (
                    product_pain_points[product].get(point, 0) + 1
                )

    # 6. Identify negative feedback by product
    if "sentiment_label" in data.columns:
        negative_data = data[
            data["sentiment_label"] == "负面"
        ].copy()
    else:
        negative_data = data[
            data["sentiment_score"] < -0.5
        ].copy()

    negative_product_counts = (
        negative_data["product"].value_counts().to_dict()
    )

    # 7. Generate targeted business recommendations
    recommendations = []

    for product in product_summary.index:

        negative_count = negative_product_counts.get(product, 0)
        issues = product_pain_points.get(product, {})

        if negative_count == 0 and not issues:
            continue

        if issues:
            top_issues = sorted(
                issues.items(),
                key=lambda item: item[1],
                reverse=True
            )[:3]

            issue_text = "、".join(
                f"{point}（{count}次）"
                for point, count in top_issues
            )

            recommendations.append(
                f"**{product}：** 当前识别到 {negative_count} 条负面反馈，"
                f"主要涉及{issue_text}。建议针对这些问题开展产品功能"
                "与使用体验排查，并通过后续用户反馈验证改进效果。"
            )

        elif negative_count > 0:
            recommendations.append(
                f"**{product}：** 当前识别到 {negative_count} 条负面反馈，"
                "建议结合原始评论进一步核查具体问题。"
            )

    if pain_point_counts:
        top_pain = max(
            pain_point_counts.items(),
            key=lambda item: item[1]
        )

        recommendations.append(
            f"**整体关注重点：** {top_pain[0]}，"
            f"在当前样本中出现 {top_pain[1]} 次，"
            "建议持续跟踪该问题的发生频率及改善情况。"
        )

    if not recommendations:
        recommendations.append(
            "当前样本未识别出明显的负面问题，"
            "建议持续收集反馈并跟踪产品表现。"
        )

    # 8. Build report
    lines = [
        "# InsightFlow 商业洞察分析报告",
        "",
        f"**生成时间：** {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## 一、消费者整体情感表现",
        "",
        f"- 总反馈数量：{total} 条",
        f"- 正面反馈：{positive} 条（{positive / total * 100:.1f}%）",
        f"- 负面反馈：{negative} 条（{negative / total * 100:.1f}%）",
        f"- 中性反馈：{neutral} 条（{neutral / total * 100:.1f}%）",
        f"- 无法判断：{unknown} 条",
        "",
        "## 二、不同产品表现分析",
        "",
    ]

    product_rows = []

    for product, row in product_summary.iterrows():
        product_rows.append([
            product,
            int(row["feedback_count"]),
            f"{row['average_rating']:.2f}"
            if pd.notna(row["average_rating"]) else "—",
            f"{row['average_sentiment']:.2f}"
            if pd.notna(row["average_sentiment"]) else "—"
        ])

    lines.extend(_markdown_table(
        ["产品", "反馈数量", "平均评分", "平均情感得分"],
        product_rows
    ))

    # 9. Consumer concerns
    lines.extend([
        "",
        "## 三、消费者关注领域",
        "",
        "注：采用多标签分类，同一条反馈可以涉及多个类别；"
        "数量按涉及该类别的反馈条数统计。",
        ""
    ])

    category_rows = [
        [category, count, f"{count / total * 100:.1f}%"]
        for category, count in sorted(
            category_counts.items(),
            key=lambda item: item[1],
            reverse=True
        )
    ]

    if category_rows:
        lines.extend(_markdown_table(
            ["关注类别", "反馈数量", "占总反馈比例"],
            category_rows
        ))
    else:
        lines.append("暂无分类数据。")

    # 10. Pain-point details
    lines.extend([
        "",
        "## 四、消费者主要痛点",
        ""
    ])

    pain_rows = []

    for product in product_summary.index:
        issues = product_pain_points.get(product, {})

        for point, count in sorted(
            issues.items(),
            key=lambda item: item[1],
            reverse=True
        ):
            pain_rows.append([product, point, count])

    if pain_rows:
        lines.extend(_markdown_table(
            ["所属产品", "消费者痛点", "出现次数"],
            pain_rows
        ))
    else:
        lines.append("当前未识别出明确的消费者痛点。")

    # 11. Business recommendations
    lines.extend([
        "",
        "## 五、商业优化建议",
        ""
    ])

    for index, item in enumerate(recommendations, 1):
        lines.append(f"{index}. {item}")

    # 12. Analytical methodology and limitations
    lines.extend([
        "",
        "## 六、分析说明与数据局限性",
        "",
        "1. 本报告整合原始消费者反馈、情感分析和文本分类结果生成。",
        "2. 情感得分反映基于当前词典规则识别的文本倾向，"
        "不等同于消费者满意度的直接测量。",
        "3. 消费者关注领域采用多标签统计，同一条反馈可能被计入多个类别。",
        "4. 痛点统计依据预设关键词匹配，可能存在语义遗漏或误识别。",
        "5. 产品平均情感得分为对应反馈得分的算术平均值，"
        "不代表经过统计检验的产品总体差异。",
        "6. 当前结论仅反映已收集样本，不能直接推广至全部消费者。",
        "7. 商业建议需要结合更大规模的数据、"
        "实际业务场景及后续验证进一步完善。",
        "",
        "---",
        "",
        "*Generated by InsightFlow*",
    ])

    # 13. Save report
    content = "\n".join(lines)

    file_path = output_path / filename

    file_path.write_text(
        content,
        encoding="utf-8"
    )

    return str(file_path)
