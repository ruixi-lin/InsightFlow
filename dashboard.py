from pathlib import Path
import html
import re

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_feedback_data, get_data_overview
from src.data_cleaner import clean_feedback_data
from src.text_analyzer import analyze_feedback_data
from src.sentiment_analyzer import analyze_sentiment
from src.llm_analyzer import analyze_with_llm, is_llm_configured


# =========================================================
# 1. 页面配置与全局样式
# =========================================================
st.set_page_config(
    page_title="InsightFlow | Consumer Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_FILE = PROJECT_ROOT / "data" / "raw" / "sports_feedback.csv"

COLORS = {
    "navy": "#17345C",
    "blue": "#397DB8",
    "teal": "#55A6A1",
    "orange": "#E47B60",
    "gray": "#91A0B3",
    "light": "#F5F7FB",
    "text": "#243247",
    "muted": "#718096",
}

st.markdown(
    """
    <style>
    :root { --ink: #243247; --muted: #718096; --line: #E7ECF3; }
    .stApp { background: #F6F8FC; color: var(--ink); }
    .block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px; }
    h1, h2, h3 { color: #17345C; letter-spacing: -0.02em; }
    h1 { font-weight: 750; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E7ECF3;
        border-radius: 14px; padding: 17px 19px;
        box-shadow: 0 2px 8px rgba(23, 52, 92, 0.035);
    }
    [data-testid="stMetricLabel"] { color: #718096; font-size: 0.88rem; }
    [data-testid="stMetricValue"] { color: #17345C; font-weight: 700; }
    section[data-testid="stSidebar"] { background: #17345C; }
    section[data-testid="stSidebar"] * { color: #F7FAFF; }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.08); border-radius: 10px; padding: 8px;
    }
    .subtitle { color: #718096; font-size: 1rem; margin-top: -0.55rem; }
    .eyebrow { color: #397DB8; font-size: 0.76rem; font-weight: 700;
               letter-spacing: 0.12em; text-transform: uppercase; }
    .insight-card {
        background: #FFFFFF; border: 1px solid #E7ECF3; border-radius: 13px;
        padding: 16px 18px; min-height: 112px;
    }
    .insight-label { color: #718096; font-size: 0.82rem; margin-bottom: 7px; }
    .insight-value { color: #17345C; font-size: 1.35rem; font-weight: 700; }
    .insight-note { color: #718096; font-size: 0.82rem; margin-top: 5px; }
    .section-note { color: #718096; font-size: 0.88rem; margin-top: -0.45rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #E7ECF3; border-radius: 10px; }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 9px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def split_tags(value):
    """Split multi-label fields using common Chinese and English separators."""
    if pd.isna(value):
        return []
    parts = re.split(r"[,，、;；|｜]", str(value))
    return [part.strip() for part in parts if part.strip()]


def count_multilabel(series):
    """Count labels; one feedback may contribute to multiple categories."""
    counts = {}
    for value in series.dropna():
        for label in split_tags(value):
            counts[label] = counts.get(label, 0) + 1
    return counts


def style_figure(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, Microsoft YaHei, sans-serif", color=COLORS["text"], size=12),
        title=dict(font=dict(size=15, color=COLORS["navy"]), x=0.02, xanchor="left"),
        margin=dict(l=18, r=18, t=58, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        coloraxis_colorbar=dict(outlinewidth=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#E7ECF3")
    fig.update_yaxes(gridcolor="#EDF1F6", zeroline=False, linecolor="#E7ECF3")
    return fig


def section_heading(title, note=None):
    st.subheader(title)
    if note:
        st.markdown(f'<p class="section-note">{html.escape(str(note))}</p>', unsafe_allow_html=True)


# =========================================================
# 2. 侧边栏：数据来源与项目说明
# =========================================================
st.sidebar.title("InsightFlow")
st.sidebar.caption("AI-Powered Consumer Insight System")
st.sidebar.divider()
st.sidebar.subheader("数据来源")

data_source = st.sidebar.radio(
    "选择数据来源",
    ["使用项目示例数据", "上传自己的CSV文件"],
    help="示例数据用于体验平台功能；上传 CSV 可分析自己的消费者反馈。",
)

uploaded_file = None
if data_source == "上传自己的CSV文件":
    uploaded_file = st.sidebar.file_uploader(
        "上传消费者反馈数据", type=["csv"], help="目前支持 CSV 格式。"
    )

st.sidebar.divider()
st.sidebar.markdown("### 分析模块")
st.sidebar.markdown(
    "- 数据概览与核心指标\n"
    "- 数据质量分析\n"
    "- 消费者情感分析\n"
    "- 产品表现对比\n"
    "- 消费者关注与痛点\n"
    "- 商业优化建议"
)
st.sidebar.divider()
st.sidebar.caption("InsightFlow | Version 1.3")
st.sidebar.caption("Consumer Feedback Analytics & Business Intelligence")


# =========================================================
# 3. 页面标题
# =========================================================
header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="eyebrow">CONSUMER INTELLIGENCE PLATFORM</div>', unsafe_allow_html=True)
    st.title("InsightFlow")
    st.markdown('<p class="subtitle">消费者反馈分析与商业洞察平台</p>', unsafe_allow_html=True)
with header_right:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:right;color:#718096;font-size:0.82rem;'>"
        "DATA-DRIVEN DECISIONS<br><b style='color:#397DB8;'>VERSION 1.3</b></div>",
        unsafe_allow_html=True,
    )
st.divider()


# =========================================================
# 4. 数据读取
# =========================================================
df = None
try:
    if data_source == "使用项目示例数据":
        if not SAMPLE_FILE.exists():
            st.error("未找到示例数据，请检查 data/raw/sports_feedback.csv。")
            st.stop()
        df = load_feedback_data(SAMPLE_FILE)
        source_label = "项目示例数据"
    else:
        if uploaded_file is None:
            st.info("请先在左侧上传 CSV 文件。")
            st.stop()
        df = load_feedback_data(uploaded_file, file_name=uploaded_file.name)
        source_label = uploaded_file.name
except Exception as error:
    st.error(f"数据读取失败：{error}")
    st.stop()


# =========================================================
# 5. 数据清洗与分析
# =========================================================
try:
    cleaned_df, quality_report = clean_feedback_data(df)

    required_columns = ["feedback_id", "product", "rating", "feedback_text"]
    missing_columns = [column for column in required_columns if column not in cleaned_df.columns]
    if missing_columns:
        st.error(f"数据缺少必要字段：{', '.join(missing_columns)}")
        st.stop()
    if cleaned_df.empty:
        st.error("清洗后没有可用于分析的数据。")
        st.stop()

    classification_result = analyze_feedback_data(cleaned_df)
    sentiment_result = analyze_sentiment(cleaned_df)

    result = cleaned_df.copy().reset_index(drop=True)
    for column in ["categories", "primary_category", "matched_keywords", "pain_points"]:
        if column in classification_result.columns:
            result[column] = classification_result[column].reset_index(drop=True)
        else:
            result[column] = ""

    for column in ["sentiment_score", "sentiment_label", "positive_hits", "negative_hits"]:
        if column in sentiment_result.columns:
            result[column] = sentiment_result[column].reset_index(drop=True)
        else:
            result[column] = 0 if column != "sentiment_label" else "无法判断"

    overview = get_data_overview(df)
except Exception as error:
    st.error(f"数据分析失败：{error}")
    st.exception(error)
    st.stop()


# =========================================================
# 6. 指标计算与自动洞察
# =========================================================
total_feedback = len(result)
available_ratings = result["rating"].dropna()
average_rating = float(available_ratings.mean()) if not available_ratings.empty else 0.0
sentiment_counts = result["sentiment_label"].value_counts()
positive_count = int(sentiment_counts.get("正面", 0))
negative_count = int(sentiment_counts.get("负面", 0))
neutral_count = int(sentiment_counts.get("中性", 0))
unknown_count = int(sentiment_counts.get("无法判断", 0))
positive_rate = positive_count / total_feedback * 100 if total_feedback else 0
negative_rate = negative_count / total_feedback * 100 if total_feedback else 0

product_summary = result.groupby("product", dropna=False).agg(
    feedback_count=("feedback_id", "count"),
    average_rating=("rating", "mean"),
    average_sentiment=("sentiment_score", "mean"),
).reset_index()
product_summary[["average_rating", "average_sentiment"]] = product_summary[
    ["average_rating", "average_sentiment"]
].round(2)
product_summary = product_summary.sort_values("feedback_count", ascending=False)

category_counts = count_multilabel(result["categories"]) if "categories" in result else {}
pain_counts = count_multilabel(result["pain_points"]) if "pain_points" in result else {}

# Build evidence-based observations from the current sample.
insights = []
if negative_count:
    insights.append(f"当前样本识别出 {negative_count} 条负面反馈（{negative_rate:.1f}%），建议优先查看对应原文及产品分布。")
if pain_counts:
    top_pain, top_pain_count = max(pain_counts.items(), key=lambda item: item[1])
    insights.append(f"出现频率最高的痛点是“{top_pain}”，共 {top_pain_count} 次；可进一步核查其具体使用场景。")
if not product_summary.empty:
    valid_products = product_summary.dropna(subset=["average_rating"])
    if not valid_products.empty:
        weakest = valid_products.sort_values("average_rating").iloc[0]
        insights.append(f"当前样本中，{weakest['product']} 的平均评分相对较低（{weakest['average_rating']:.2f}/5），值得结合反馈内容排查。")
if not insights:
    insights.append("当前样本暂未呈现突出的负面信号，建议持续积累反馈并观察变化。")


# =========================================================
# 7. 核心数据概览
# =========================================================
section_heading("核心数据概览", f"当前数据来源：{source_label} · 基于清洗后 {total_feedback} 条有效记录")
metric_cols = st.columns(4)
metric_cols[0].metric("反馈总量", f"{total_feedback:,} 条")
metric_cols[1].metric("平均产品评分", f"{average_rating:.2f} / 5" if not available_ratings.empty else "暂无评分")
metric_cols[2].metric("正面反馈占比", f"{positive_rate:.1f}%")
metric_cols[3].metric("识别负面反馈", f"{negative_count} 条", delta=f"{negative_rate:.1f}% 的样本", delta_color="inverse")

st.markdown("#### 本次分析摘要")
insight_cols = st.columns(min(3, len(insights)))
for index, insight in enumerate(insights[:3]):
    with insight_cols[index]:
        st.markdown(
            f'<div class="insight-card"><div class="insight-label">KEY OBSERVATION {index + 1}</div>'
            f'<div style="color:#243247;font-size:0.94rem;line-height:1.65;">{html.escape(insight)}</div></div>',
            unsafe_allow_html=True,
        )
st.caption("以上观察由当前样本中的统计结果生成，用于辅助定位问题，不构成因果判断。")
st.divider()


# =========================================================
# 8. 消费者情感分析
# =========================================================
section_heading("消费者情感分析", "从反馈情感标签与评分分布两个角度观察消费者的总体反馈倾向。")
sentiment_col, rating_col = st.columns(2)

with sentiment_col:
    sentiment_order = ["正面", "中性", "负面", "无法判断"]
    sentiment_plot = result["sentiment_label"].value_counts().reindex(sentiment_order, fill_value=0).reset_index()
    sentiment_plot.columns = ["情感类别", "反馈数量"]
    sentiment_colors = {"正面": COLORS["teal"], "中性": COLORS["gray"], "负面": COLORS["orange"], "无法判断": "#CBD5E1"}
    fig_sentiment = px.pie(
        sentiment_plot, names="情感类别", values="反馈数量", hole=0.62,
        title="消费者情感分布", color="情感类别", color_discrete_map=sentiment_colors,
    )
    fig_sentiment.update_traces(textposition="inside", textinfo="percent", hole=0.62)
    style_figure(fig_sentiment, 360)
    st.plotly_chart(fig_sentiment, width="stretch", config={"displaylogo": False})

with rating_col:
    rating_counts = result["rating"].dropna().value_counts().sort_index().rename_axis("评分").reset_index(name="反馈数量")
    if not rating_counts.empty:
        fig_rating = px.bar(
            rating_counts, x="评分", y="反馈数量", text="反馈数量",
            title="产品评分分布", color="反馈数量", color_continuous_scale="Blues",
        )
        fig_rating.update_xaxes(dtick=1)
        fig_rating.update_traces(textposition="outside", marker_line_width=0)
        style_figure(fig_rating, 360)
        st.plotly_chart(fig_rating, width="stretch", config={"displaylogo": False})
    else:
        st.info("当前数据没有可用的评分记录。")
st.divider()


# =========================================================
# 9. 产品表现分析
# =========================================================
section_heading("不同产品表现分析", "结合反馈数量、平均评分和平均情感得分，比较不同产品在当前样本中的表现。")
product_col1, product_col2 = st.columns(2)
with product_col1:
    fig_product_rating = px.bar(
        product_summary, x="product", y="average_rating", text="average_rating",
        title="不同产品平均评分", color="average_rating", color_continuous_scale="Blues",
        hover_data={"feedback_count": True},
    )
    fig_product_rating.update_yaxes(range=[0, 5], title="平均评分")
    fig_product_rating.update_traces(textposition="outside", marker_line_width=0)
    style_figure(fig_product_rating, 360)
    st.plotly_chart(fig_product_rating, width="stretch", config={"displaylogo": False})
with product_col2:
    fig_product_sentiment = px.bar(
        product_summary, x="product", y="average_sentiment", text="average_sentiment",
        title="不同产品平均情感得分", color="average_sentiment",
        color_continuous_scale="RdYlGn", hover_data={"feedback_count": True},
    )
    fig_product_sentiment.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    fig_product_sentiment.update_yaxes(title="平均情感得分")
    fig_product_sentiment.update_traces(textposition="outside", marker_line_width=0)
    style_figure(fig_product_sentiment, 360)
    st.plotly_chart(fig_product_sentiment, width="stretch", config={"displaylogo": False})

product_display = product_summary.rename(columns={
    "product": "产品", "feedback_count": "反馈数量",
    "average_rating": "平均评分", "average_sentiment": "平均情感得分",
})
st.dataframe(product_display, width="stretch", hide_index=True)
st.caption("平均情感得分是当前情感分析规则对反馈文本打分后的均值，不等同于满意度量表得分。")
st.divider()


# =========================================================
# 10. 消费者关注领域
# =========================================================
section_heading("消费者关注领域", "采用多标签统计，同一条反馈可能涉及多个关注类别，因此各类别数量之和可能大于反馈总量。")
if category_counts:
    category_df = pd.DataFrame(list(category_counts.items()), columns=["关注类别", "反馈数量"]).sort_values("反馈数量", ascending=True)
    fig_category = px.bar(
        category_df, x="反馈数量", y="关注类别", orientation="h", text="反馈数量",
        title="消费者关注类别统计", color="反馈数量", color_continuous_scale="Blues",
    )
    fig_category.update_traces(textposition="outside", marker_line_width=0)
    style_figure(fig_category, max(300, 58 * len(category_df) + 100))
    st.plotly_chart(fig_category, width="stretch", config={"displaylogo": False})
else:
    st.info("当前数据未识别出明确的关注类别。")
st.divider()


# =========================================================
# 11. 消费者主要痛点
# =========================================================
section_heading("消费者主要痛点", "痛点词来自文本分类结果，建议结合下方原始反馈进一步判断问题的具体含义。")
if pain_counts:
    pain_df = pd.DataFrame(list(pain_counts.items()), columns=["消费者痛点", "出现次数"]).sort_values("出现次数", ascending=False)
    pain_col, pain_table_col = st.columns([1.15, 1])
    with pain_col:
        fig_pain = px.bar(
            pain_df.sort_values("出现次数", ascending=True),
            x="出现次数", y="消费者痛点", orientation="h", text="出现次数",
            title="痛点出现频次", color="出现次数", color_continuous_scale="OrRd",
        )
        fig_pain.update_traces(textposition="outside", marker_line_width=0)
        style_figure(fig_pain, max(300, 55 * len(pain_df) + 100))
        st.plotly_chart(fig_pain, width="stretch", config={"displaylogo": False})
    with pain_table_col:
        st.dataframe(pain_df, width="stretch", hide_index=True, height=300)
else:
    st.info("当前没有识别到明确的消费者痛点。")
st.divider()


# =========================================================
# 12. AI 深度消费者洞察
# =========================================================
section_heading(
    "AI 深度消费者洞察",
    "基于 DeepSeek 大语言模型，深入识别消费者情感、产品痛点、显性需求、隐含需求及优化建议。"
)

# Initialize AI records before the optional AI module
ai_records = []

if not is_llm_configured():
    st.warning("尚未配置 DeepSeek API Key，请检查 .env 文件。")
else:
    # 使用 Session State 缓存，避免页面刷新后重复调用 API
    if "ai_analysis_cache" not in st.session_state:
        st.session_state["ai_analysis_cache"] = {}

    ai_limit = st.number_input(
        "本次最多分析多少条反馈",
        min_value=1,
        max_value=max(1, len(result)),
        value=min(10, max(1, len(result))),
        step=1,
        key="ai_analysis_limit"
    )

    st.caption(
        f"当前共有 {len(result)} 条有效反馈。"
        "AI 分析将按需调用 API，已分析的反馈会优先使用缓存。"
    )

    if st.button(
        "启动 AI 深度分析",
        type="primary",
        key="run_ai_analysis"
    ):
        cache = st.session_state["ai_analysis_cache"]
        selected_data = result.head(int(ai_limit))

        progress = st.progress(0)
        status_text = st.empty()

        success_count = 0
        error_count = 0

        for index, row in enumerate(
            selected_data.itertuples(index=False),
            start=1
        ):
            feedback_id = str(getattr(row, "feedback_id", index))
            feedback_text = str(row.feedback_text).strip()
            product = str(row.product)

            cache_key = (
                feedback_id,
                product,
                feedback_text
            )

            if not feedback_text or feedback_text.lower() == "nan":
                error_count += 1
                status_text.write(
                    f"正在处理 {index}/{len(selected_data)}："
                    "跳过空白反馈"
                )
                progress.progress(index / len(selected_data))
                continue

            if cache_key not in cache:
                try:
                    response = analyze_with_llm(feedback_text)

                    if response.get("status") != "success":
                        raise ValueError("AI 分析未成功返回")

                    cache[cache_key] = {
                        "status": "success",
                        "analysis": response["analysis"],
                        "model": response.get("model", "DeepSeek")
                    }

                except Exception as error:
                    cache[cache_key] = {
                        "status": "error",
                        "message": str(error)
                    }

            if cache[cache_key]["status"] == "success":
                success_count += 1
            else:
                error_count += 1

            status_text.write(
                f"正在分析：{index}/{len(selected_data)}"
            )
            progress.progress(index / len(selected_data))

        status_text.success(
            f"分析完成！成功 {success_count} 条，"
            f"失败 {error_count} 条。"
        )

    # 展示当前数据中已有的 AI 分析结果
    cache = st.session_state["ai_analysis_cache"]
    ai_records = []

    for row in result.itertuples(index=False):
        feedback_id = str(getattr(row, "feedback_id", ""))
        product = str(row.product)
        feedback_text = str(row.feedback_text).strip()

        cache_key = (
            feedback_id,
            product,
            feedback_text
        )

        cached = cache.get(cache_key)

        if not cached or cached["status"] != "success":
            continue

        analysis = cached["analysis"]

        ai_records.append({
            "反馈ID": feedback_id,
            "产品": product,
            "消费者原始反馈": feedback_text,
            "情感判断": analysis["sentiment"],
            "正面评价": "；".join(analysis["positive_aspects"]),
            "负面评价": "；".join(analysis["negative_aspects"]),
            "消费者痛点": "；".join(analysis["pain_points"]),
            "显性需求": "；".join(analysis["explicit_needs"]),
            "隐含需求": "；".join(analysis["inferred_needs"]),
            "AI优化建议": "；".join(analysis["suggestions"]),
            "置信度": analysis["confidence"]
        })

    if ai_records:
        ai_df = pd.DataFrame(ai_records)

        st.success(f"目前已完成 {len(ai_records)} 条反馈的 AI 分析。")

        st.dataframe(
            ai_df,
            width="stretch",
            hide_index=True,
            height=420
        )

        # 单条反馈详情
        st.markdown("### 单条反馈深度解读")

        selected_id = st.selectbox(
            "选择一条反馈查看 AI 分析详情",
            ai_df["反馈ID"].tolist(),
            key="ai_detail_selection"
        )

        detail = ai_df[
            ai_df["反馈ID"] == selected_id
        ].iloc[0]

        st.markdown(f"**原始反馈：** {detail['消费者原始反馈']}")
        st.markdown(f"**AI 情感判断：** {detail['情感判断']}")
        st.markdown(f"**消费者痛点：** {detail['消费者痛点'] or '未识别到明确痛点'}")
        st.markdown(f"**显性需求：** {detail['显性需求'] or '无'}")
        st.markdown(f"**隐含需求：** {detail['隐含需求'] or '无'}")
        st.markdown(f"**优化建议：** {detail['AI优化建议'] or '暂无建议'}")

    else:
        st.info("尚无 AI 分析结果，请点击上方按钮启动分析。")

st.divider()

# =========================================================
# 12. 商业优化建议
# =========================================================
section_heading("商业优化建议", "建议根据反馈频次与影响范围确定核查优先级；小样本结果应先验证，再决定是否调整产品。")
recommendations = []
if negative_count > 0:
    recommendations.append(("优先核查负面反馈", f"当前识别到 {negative_count} 条负面反馈。建议先查看对应产品、评分和原文，区分功能问题、质量问题与使用体验问题。"))
if pain_counts:
    top_pain, top_pain_count = max(pain_counts.items(), key=lambda item: item[1])
    recommendations.append((f"跟进高频痛点：{top_pain}", f"该痛点在当前样本中出现 {top_pain_count} 次。建议回看原始语境，确认是否集中于某一产品或使用环节。"))
if not product_summary.empty:
    valid_products = product_summary.dropna(subset=["average_rating"])
    if not valid_products.empty:
        weakest = valid_products.sort_values("average_rating").iloc[0]
        recommendations.append((f"复核产品：{weakest['product']}", f"该产品当前平均评分为 {weakest['average_rating']:.2f}/5。建议结合反馈量及负面反馈内容，判断是否需要专项排查。"))
if not recommendations:
    recommendations.append(("持续跟踪消费者反馈", "当前样本未识别出明显的负面集中问题。建议持续收集反馈，并按产品和时间维度观察变化。"))

for idx, (title, description) in enumerate(recommendations, start=1):
    with st.container(border=True):
        st.markdown(f"**{idx:02d} · {title}**")
        st.write(description)

st.caption("以上建议为基于样本统计特征生成的核查方向，并非自动验证过的产品问题或因果结论。")
st.divider()


# =========================================================
# 13. 数据质量分析
# =========================================================
section_heading("数据质量分析", "展示原始数据清洗前后的记录变化，以及识别到的异常类型。")
quality_col1, quality_col2, quality_col3 = st.columns(3)
quality_col1.metric("原始数据量", f"{quality_report['original_rows']:,} 条")
quality_col2.metric("清洗后数据量", f"{quality_report['cleaned_rows']:,} 条")
quality_col3.metric("数据保留率", f"{quality_report['data_retention_rate']:.2f}%")
quality_a, quality_b = st.columns(2)
quality_a.caption(f"完全重复记录移除：{quality_report['duplicates_removed']} 条")
quality_b.caption(f"识别异常问题：{quality_report['anomaly_count']} 项")

with st.expander("查看数据质量详情"):
    quality_summary = {key: value for key, value in quality_report.items() if key != "anomaly_records"}
    st.json(quality_summary)
    if quality_report.get("anomaly_records"):
        anomaly_df = pd.DataFrame(quality_report["anomaly_records"])
        st.markdown("**异常记录明细**")
        st.dataframe(anomaly_df, width="stretch", hide_index=True)
    else:
        st.success("本次清洗未识别到需要记录的异常问题。")

with st.expander("查看原始数据概览"):
    overview_cols = st.columns(4)
    overview_cols[0].metric("字段数量", overview.get("total_columns", 0))
    overview_cols[1].metric("原始缺失值", overview.get("missing_values", 0))
    overview_cols[2].metric("原始完全重复行", overview.get("duplicate_records", 0))
    overview_cols[3].metric("内存占用", f"{overview.get('memory_usage_mb', 0):.2f} MB")
    st.write("字段列表：" + "、".join(overview.get("columns", [])))
st.divider()


# =========================================================
# 14. 数据预览与下载
# =========================================================

section_heading("消费者反馈数据", "可查看传统文本分析与 AI 深度分析后的完整数据，并下载 CSV 用于后续核查。")

st.dataframe(result, width="stretch", hide_index=True, height=420)

# 整合传统分析结果与 AI 深度分析结果
export_result = result.copy()

if ai_records:
    ai_export_df = pd.DataFrame(ai_records).rename(columns={
        "反馈ID": "feedback_id",
        "产品": "product",
        "消费者原始反馈": "feedback_text",
    })

    export_result = export_result.merge(
        ai_export_df,
        on=["feedback_id", "product", "feedback_text"],
        how="left"
    )

csv_data = export_result.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="下载完整分析结果 CSV",
    data=csv_data,
    file_name="insightflow_analysis_results.csv",
    mime="text/csv",
    type="primary",
)



# =========================================================
# 15. 页脚
# =========================================================
st.divider()
footer_left, footer_right = st.columns([3, 1])
with footer_left:
    st.caption("InsightFlow | Consumer Feedback Analytics and Business Intelligence")
with footer_right:
    st.caption("Version 1.3 · Built with Streamlit")
