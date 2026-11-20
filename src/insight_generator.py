
from pathlib import Path
from datetime import datetime
import json


def generate_data_quality_report(
    report,
    output_dir="outputs/reports",
    filename="data_quality_report.md",
):
    """
    Generate a structured data quality report.

    Parameters
    ----------
    report : dict
        Data quality report from clean_feedback_data().
    output_dir : str
        Directory for saving the report.
    filename : str
        Output Markdown filename.

    Returns
    -------
    str
        Generated report file path.
    """

    if not isinstance(report, dict):
        raise TypeError("Report must be a dictionary.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Extract quality metrics
    original_rows = report.get("original_rows", 0)
    cleaned_rows = report.get("cleaned_rows", 0)
    retention_rate = report.get("data_retention_rate", 0)

    anomaly_records = report.get("anomaly_records", [])
    anomaly_summary = report.get("anomaly_summary", {})

    anomaly_count = report.get(
        "anomaly_count",
        len(anomaly_records)
    )

    # 2. Define anomaly labels
    anomaly_labels = {
        "exact_duplicate": "完全重复记录",
        "duplicate_feedback_id": "重复反馈ID",
        "invalid_rating": "无效评分",
        "invalid_date": "无效日期",
        "missing_feedback": "缺失反馈文本",
        "missing_feedback_id": "缺失反馈ID",
    }

    # 3. Build report content
    lines = [
        "# InsightFlow 数据质量分析报告",
        "",
        f"**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 一、数据质量概览",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 原始数据量 | {original_rows} 条 |",
        f"| 清洗后数据量 | {cleaned_rows} 条 |",
        f"| 完全重复记录移除 | {report.get('duplicates_removed', 0)} 条 |",
        f"| 重复反馈ID | {report.get('duplicate_feedback_ids', 0)} 条 |",
        f"| 清洗前缺失值 | {report.get('missing_values_before', 0)} 个 |",
        f"| 标准化后缺失值 | {report.get('missing_after_standardization', 0)} 个 |",
        f"| 清洗后缺失值 | {report.get('missing_values_after', 0)} 个 |",
        f"| 无效评分 | {report.get('invalid_ratings', 0)} 条 |",
        f"| 无效日期 | {report.get('invalid_dates', 0)} 条 |",
        f"| 缺失反馈文本 | {report.get('missing_feedback', 0)} 条 |",
        f"| 缺失反馈ID | {report.get('missing_feedback_ids', 0)} 条 |",
        f"| 数据保留率 | {retention_rate}% |",
        "",
        "## 二、异常记录分析",
        "",
        f"本次共识别 **{anomaly_count} 项异常问题**。",
        "",
        "| 异常类型 | 数量 |",
        "|---|---:|",
    ]

    for issue_type, count in anomaly_summary.items():
        label = anomaly_labels.get(issue_type, issue_type)
        lines.append(f"| {label} | {count} |")

    if not anomaly_summary:
        lines.append("| 暂无异常分类数据 | 0 |")

    # 4. Add anomaly details
    lines.extend([
        "",
        "## 三、异常记录明细",
        "",
        "| 来源行 | 反馈ID | 异常类型 | 原始值 | 问题说明 |",
        "|---:|---|---|---|---|",
    ])

    if anomaly_records:
        for item in anomaly_records:
            source_row = item.get("source_row", "-")
            feedback_id = item.get("feedback_id") or "缺失"
            issue_type = item.get("issue_type", "unknown")
            label = anomaly_labels.get(issue_type, issue_type)

            original_value = item.get("original_value", "")
            if isinstance(original_value, (dict, list)):
                original_value = json.dumps(
                    original_value,
                    ensure_ascii=False,
                    default=str,
                )

            details = str(item.get("details", ""))

            # Prevent Markdown table formatting issues
            feedback_id = str(feedback_id).replace("|", "\\|").replace("\n", " ")
            original_value = str(original_value).replace("|", "\\|").replace("\n", " ")
            details = details.replace("|", "\\|").replace("\n", " ")

            lines.append(
                f"| {source_row} | {feedback_id} | "
                f"{label} | {original_value} | {details} |"
            )
    else:
        lines.append("| - | - | 无异常 | - | 数据质量正常 |")

    # 5. Generate recommendations
    recommendations = []
    if report.get("duplicates_removed", 0) > 0:
        recommendations.append(
            "检查完全重复记录，确认是否因重复采集或重复导入造成，并完善去重机制。"
        )

    if report.get("duplicate_feedback_ids", 0) > 0:
        recommendations.append(
            "检查重复反馈ID，确认是否存在重复提交或数据标识冲突。"
        )

    if report.get("invalid_ratings", 0) > 0:
        recommendations.append(
            "核查评分字段，确保评分符合1至5分的有效范围。"
        )

    if report.get("invalid_dates", 0) > 0:
        recommendations.append(
            "检查日期格式，统一日期字段的记录标准。"
        )

    if report.get("missing_feedback", 0) > 0:
        recommendations.append(
            "检查缺失的反馈文本，必要时联系数据来源补充。"
        )

    if report.get("missing_feedback_ids", 0) > 0:
        recommendations.append(
            "检查缺失反馈ID的记录，补充唯一标识以便追踪。"
        )

    if not recommendations:
        recommendations.append(
            "当前未发现上述类型的质量问题，建议继续进行常规质量监测。"
        )

    lines.extend([
        "",
        "## 四、数据质量改进建议",
        "",
    ])

    for index, recommendation in enumerate(
        recommendations, start=1
    ):
        lines.append(f"{index}. {recommendation}")

    # 6. Add conclusion
    lines.extend([
        "",
        "## 五、总结",
        "",
        f"本次数据处理共检查 {original_rows} 条原始记录，"
        f"清洗后保留 {cleaned_rows} 条，"
        f"数据保留率为 {retention_rate}%。",
        "",
        "异常记录已单独整理，可结合原始数据进行进一步核查。",
        "",
        "---",
        "",
        "*Generated by InsightFlow*",
    ])

    # 7. Save report
    content = "\n".join(lines)

    file_path = output_path / filename
    file_path.write_text(content, encoding="utf-8")

    return str(file_path)
