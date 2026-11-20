
import re
import pandas as pd
from collections import Counter


# =========================================================
# 1. Feedback categories
# =========================================================

CATEGORY_KEYWORDS = {
    "产品质量": [
        "质量", "做工", "漏水", "开裂", "损坏",
        "破损", "故障", "耐用", "松动", "异响"
    ],
    "功能性能": [
        "容量", "密封", "音质", "续航", "降噪",
        "连接", "运动数据", "记录准确", "功能",
        "性能", "充电", "电池", "灵敏度"
    ],
    "使用体验": [
        "佩戴", "操作", "携带", "清洗", "使用体验",
        "舒适", "不舒服", "方便", "复杂", "重量",
        "容易使用", "手感"
    ],
    "外观设计": [
        "外观", "颜色", "设计", "颜值", "好看",
        "美观", "款式", "漂亮"
    ],
    "物流包装": [
        "物流", "快递", "配送", "包装", "发货",
        "运输", "送货"
    ],
    "价格与售后": [
        "价格", "价钱", "性价比", "便宜", "昂贵",
        "退款", "退货", "客服", "售后", "维修",
        "换货", "贵"
    ],
}


# =========================================================
# 2. Pain-point keywords
# =========================================================

PAIN_POINT_KEYWORDS = [
    "漏水", "开裂", "损坏", "故障",
    "连接断开", "断开", "连接不稳定",
    "不舒服", "不理想", "不好用",
    "操作复杂", "复杂",
    "不方便", "不太方便",
    "续航短", "续航太差","续航也太差", "耗电快",
    "音质差", "噪音大",
    "质量差", "做工差",
    "价格贵", "太贵", "有点贵",
    "性价比低", "物流慢", "包装破损",
    "无法使用", "容易坏",
    "速度太慢", "响应太慢",
    "一点用都没有", "毫无用处"
]


# =========================================================
# 3. Explicit needs
# =========================================================

NEED_KEYWORDS = {
    "提升产品质量": [
        "质量差", "容易坏", "开裂", "损坏",
        "做工差", "故障"
    ],
    "改善功能性能": [
        "续航短", "续航太差", "耗电快",
        "连接不稳定", "音质差", "降噪差", "续航也太差",
    ],
    "优化使用体验": [
        "操作复杂", "不方便", "不舒服",
        "清洗不方便", "佩戴不舒服"
    ],
    "改善价格与售后": [
        "价格贵", "太贵", "性价比低",
        "客服态度差", "响应太慢"
    ],
    "改善物流包装": [
        "物流慢", "包装破损", "配送慢"
    ],
}


# =========================================================
# 4. Negation expressions
# =========================================================

NEGATION_PATTERNS = [
    "没有", "并没有", "并非",
    "不是", "并不", "不", "没", "无", "未"
]


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def _is_negated(text, keyword):
    """Check whether a keyword is preceded by a nearby negation."""

    position = text.find(keyword)

    if position < 0:
        return False

    prefix = text[max(0, position - 8):position]

    return any(
        prefix.endswith(negation)
        for negation in NEGATION_PATTERNS
    )


# =========================================================
# 5. Analyze one feedback record
# =========================================================

def analyze_feedback_text(text):

    text = preprocess_text(text)

    if not text:
        return {
            "categories": [],
            "primary_category": "未分类",
            "matched_keywords": [],
            "pain_points": [],
            "negated_issues": [],
            "pain_evidence": [],
            "explicit_needs": [],
            "review_status": "待复核",
        }

    matched_categories = []
    matched_keywords = []

    # Multilabel classification
    for category, keywords in CATEGORY_KEYWORDS.items():

        current_matches = [
            keyword
            for keyword in keywords
            if keyword in text
        ]

        if current_matches:
            matched_categories.append(category)
            matched_keywords.extend(current_matches)

    
    # Detect actual and negated pain points
    pain_points = []
    negated_issues = []

    matched_pain_keywords = [
        keyword
        for keyword in PAIN_POINT_KEYWORDS
        if keyword in text
    ]

    # Remove shorter keywords already covered by longer phrases.
    filtered_pain_keywords = [
        keyword
        for keyword in matched_pain_keywords
        if not any(
            keyword != other and keyword in other
            for other in matched_pain_keywords
        )
    ]

    for keyword in filtered_pain_keywords:

        if _is_negated(text, keyword):
            negated_issues.append(keyword)
        else:
            pain_points.append(keyword)

    # Extract and merge overlapping pain evidence.
    pain_spans = []

    for keyword in pain_points:

        for match in re.finditer(re.escape(keyword), text):
            pain_spans.append(
                (match.start(), match.end())
            )

    pain_spans.sort()

    merged_spans = []

    for start, end in pain_spans:

        if merged_spans and start <= merged_spans[-1][1]:
            merged_spans[-1] = (
                merged_spans[-1][0],
                max(merged_spans[-1][1], end)
            )
        else:
            merged_spans.append((start, end))

    # Merge overlapping evidence context windows.
    evidence_spans = []

    for start, end in merged_spans:

        context_start = max(0, start - 12)
        context_end = min(len(text), end + 12)

        if evidence_spans and context_start <= evidence_spans[-1][1]:

            evidence_spans[-1] = (
                evidence_spans[-1][0],
                max(evidence_spans[-1][1], context_end)
            )

        else:
            evidence_spans.append(
                (context_start, context_end)
            )

    pain_evidence = [
        text[start:end]
        for start, end in evidence_spans
    ]

    # Extract explicit needs.
    explicit_needs = []

    for need, keywords in NEED_KEYWORDS.items():

        if any(
            keyword in text
            and not _is_negated(text, keyword)
            for keyword in keywords
        ):
            explicit_needs.append(need)

    # Determine primary category.
    primary_category = (
        matched_categories[0]
        if matched_categories
        else "其他"
    )

    # Mark complex or potentially ambiguous expressions.
    ambiguous_expressions = [
        "不是不好",
        "不是不满意",
        "虽然",
        "但是",
        "不过",
        "所谓",
        "一点也不",
    ]

    review_status = (
        "待复核"
        if any(expr in text for expr in ambiguous_expressions)
        else "自动分析"
    )

    return {
        "categories": matched_categories,
        "primary_category": primary_category,
        "matched_keywords": list(
            dict.fromkeys(matched_keywords)
        ),
        "pain_points": list(
            dict.fromkeys(pain_points)
        ),
        "negated_issues": list(
            dict.fromkeys(negated_issues)
        ),
        "pain_evidence": pain_evidence,
        "explicit_needs": explicit_needs,
        "review_status": review_status,
    }


# =========================================================
# 6. Analyze an entire dataset
# =========================================================

def analyze_feedback_data(
    df,
    feedback_column="feedback_text"
):

    if df is None or df.empty:
        raise ValueError("Input dataset is empty.")

    if feedback_column not in df.columns:
        raise KeyError(
            f"Column '{feedback_column}' not found."
        )

    result = df.copy()

    analysis_results = result[feedback_column].apply(
        analyze_feedback_text
    )

    result["categories"] = analysis_results.apply(
        lambda x: "、".join(x["categories"])
    )

    result["primary_category"] = analysis_results.apply(
        lambda x: x["primary_category"]
    )

    result["matched_keywords"] = analysis_results.apply(
        lambda x: "、".join(x["matched_keywords"])
    )

    result["pain_points"] = analysis_results.apply(
        lambda x: "、".join(x["pain_points"])
    )

    result["negated_issues"] = analysis_results.apply(
        lambda x: "、".join(x["negated_issues"])
    )

    result["pain_evidence"] = analysis_results.apply(
        lambda x: "；".join(x["pain_evidence"])
    )

    result["explicit_needs"] = analysis_results.apply(
        lambda x: "、".join(x["explicit_needs"])
    )

    result["review_status"] = analysis_results.apply(
        lambda x: x["review_status"]
    )

    return result


# =========================================================
# 7. Summarize category distribution
# =========================================================

def summarize_categories(
    analyzed_df,
    category_column="categories"
):

    if category_column not in analyzed_df.columns:
        raise KeyError(
            f"Column '{category_column}' not found."
        )

    category_counter = Counter()

    for categories in analyzed_df[category_column].dropna():

        for category in str(categories).split("、"):
            if category:
                category_counter[category] += 1

    return pd.DataFrame([
        {
            "category": category,
            "feedback_count": count,
            "percentage": round(
                count / len(analyzed_df) * 100, 2
            )
        }
        for category, count in category_counter.most_common()
    ])


# =========================================================
# 8. Summarize recurring pain points
# =========================================================

def summarize_pain_points(
    analyzed_df,
    pain_column="pain_points"
):

    if pain_column not in analyzed_df.columns:
        raise KeyError(
            f"Column '{pain_column}' not found."
        )

    pain_counter = Counter()

    for points in analyzed_df[pain_column].dropna():

        for point in str(points).split("、"):
            if point:
                pain_counter[point] += 1

    return pd.DataFrame([
        {
            "pain_point": point,
            "feedback_count": count
        }
        for point, count in pain_counter.most_common()
    ])
