
import re
import pandas as pd


# =========================================================
# 1. Sentiment lexicons
# =========================================================

POSITIVE_WORDS = {
    "非常满意": 2.0,
    "体验很好": 2.0,
    "值得推荐": 2.0,
    "非常好": 2.0,
    "很方便": 1.5,
    "很满意": 1.5,
    "很好": 1.5,
    "不错": 1.0,
    "准确": 1.0,
    "合适": 1.0,
    "方便": 1.0,
    "好看": 1.0,
    "漂亮": 1.0,
    "喜欢": 1.0,
    "满意": 1.5,
    "优秀": 1.5,
    "稳定": 1.0,
    "清晰": 1.0,
    "舒适": 1.0,
    "推荐": 1.0,
    "完美": 2.0,
}

NEGATIVE_WORDS = {
    "非常失望": -2.0,
    "体验很差": -2.0,
    "经常断开": -2.0,
    "无法使用": -2.0,
    "清洗不太方便": -1.5,
    "不方便清洗": -1.5,
    "解决速度太慢": -2.0,
    "一点用都没有": -2.0,
    "没有想象中那么好": -1.5,
    "续航太差": -2.0,
    "续航不足": -1.5,
    "续航短": -1.5,
    "不满意": -1.5,
    "不舒服": -1.5,
    "不理想": -1.5,
    "不好用": -1.5,
    "不方便": -1.5,
    "不太方便": -1.5,
    "不喜欢": -1.5,
    "不稳定": -1.5,
    "不智能": -1.5,
    "太差": -1.8,
    "太贵": -2.0,
    "有点贵": -1.5,
    "价格贵": -1.5,
    "昂贵": -1.5,
    "响应太慢": -1.5,
    "毫无用处": -2.0,
    "没有用": -1.5,
    "功能失效": -2.0,
    "无法解决": -1.5,
    "耗电快": -1.5,
    "漏水": -1.5,
    "断开": -1.5,
    "复杂": -1.0,
    "不适": -1.0,
    "问题": -1.0,
    "失望": -1.5,
    "差": -1.0,
    "困难": -1.0,
    "噪音": -1.0,
    "费劲": -1.0,
}

NEGATION_WORDS = [
    "并非", "不是", "并不",
    "没有", "不", "没", "无", "未"
]

INTENSIFIERS = {
    "非常": 1.5,
    "特别": 1.5,
    "十分": 1.5,
    "很": 1.2,
    "比较": 1.1,
    "有点": 0.8,
    "稍微": 0.7,
    "太": 1.5,
}

CONTRAST_WORDS = [
    "但是", "不过", "然而",
    "可是", "就是", "但"
]


# =========================================================
# 2. Split feedback into contrast segments
# =========================================================

def _split_segments(text):
    pattern = "(" + "|".join(
        re.escape(word)
        for word in sorted(
            CONTRAST_WORDS,
            key=len,
            reverse=True
        )
    ) + ")"

    parts = re.split(pattern, text)

    segments = []
    after_contrast = False

    for part in parts:
        if not part:
            continue

        if part in CONTRAST_WORDS:
            after_contrast = True
            continue

        segments.append({
            "text": part,
            "weight": 1.5 if after_contrast else 1.0
        })

    return segments


# =========================================================
# 3. Analyze one text segment
# =========================================================

def _match_segment(text):

    words = {
        **POSITIVE_WORDS,
        **NEGATIVE_WORDS
    }

    pattern = "|".join(
        re.escape(word)
        for word in sorted(
            words,
            key=len,
            reverse=True
        )
    )

    matches = list(re.finditer(pattern, text))

    score = 0.0
    positive_hits = []
    negative_hits = []
    negated_hits = []

    occupied_until = -1

    for match in matches:

        if match.start() < occupied_until:
            continue

        word = match.group()
        value = words[word]

        prefix = text[:match.start()]

        # Avoid interpreting "not bad" as a simple negative.
        double_negation = re.search(
            r"(不是|并非|并不)\s*$",
            prefix
        )

        if double_negation and word in [
            "不好", "不满意", "不理想"
        ]:
            positive_hits.append(word)
            score += abs(value) * 0.5
            occupied_until = match.end()
            continue

        # Detect local negation.
        negated = re.search(
            r"(并非|不是|并不|没有|不|没|无|未)"
            r"[^，。！？；、]{0,3}$",
            prefix
        )

        if negated:

            # Negated negative expression:
            # record it as an absence of a problem.
            if value < 0:
                negated_hits.append(word)
                occupied_until = match.end()
                continue

            # Negated positive expression:
            # treat as negative sentiment.
            value = -value

        # Apply the nearest intensity modifier.
        local_prefix = prefix[-6:]

        for intensity, factor in sorted(
            INTENSIFIERS.items(),
            key=lambda item: len(item[0]),
            reverse=True
        ):
            if local_prefix.endswith(intensity):
                value *= factor
                break

        score += value

        if value > 0:
            positive_hits.append(word)
        elif value < 0:
            negative_hits.append(word)

        occupied_until = match.end()

    return (
        score,
        positive_hits,
        negative_hits,
        negated_hits
    )


# =========================================================
# 4. Analyze one feedback record
# =========================================================

def analyze_single_sentiment(text):

    if pd.isna(text) or not str(text).strip():
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "无法判断",
            "positive_hits": "",
            "negative_hits": "",
            "negated_hits": "",
            "review_status": "待复核",
        }

    text = str(text).strip()

    total_score = 0.0
    positive_hits = []
    negative_hits = []
    negated_hits = []

    for segment in _split_segments(text):

        score, pos, neg, negated = _match_segment(
            segment["text"]
        )

        total_score += score * segment["weight"]

        positive_hits.extend(pos)
        negative_hits.extend(neg)
        negated_hits.extend(negated)

    total_score = round(total_score, 2)

    # Determine sentiment label.
    if positive_hits and negative_hits:
        label = "混合"
    elif total_score > 0.5:
        label = "正面"
    elif total_score < -0.5:
        label = "负面"
    else:
        label = "中性"

    # Mark potentially ambiguous language.
    ambiguous_patterns = [
        "不是不好",
        "不是不满意",
        "并非不好",
        "没有想象中那么好",
        "一点用都没有",
        "所谓",
    ]

    needs_review = any(
        phrase in text
        for phrase in ambiguous_patterns
    )

    review_status = (
        "待复核" if needs_review else "自动分析"
    )

    return {
        "sentiment_score": total_score,
        "sentiment_label": label,
        "positive_hits": "、".join(
            dict.fromkeys(positive_hits)
        ),
        "negative_hits": "、".join(
            dict.fromkeys(negative_hits)
        ),
        "negated_hits": "、".join(
            dict.fromkeys(negated_hits)
        ),
        "review_status": review_status,
    }


# =========================================================
# 5. Analyze an entire feedback dataset
# =========================================================

def analyze_sentiment(
    df,
    text_column="feedback_text"
):

    if df is None or df.empty:
        raise ValueError("Input dataset is empty.")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' not found."
        )

    result = df.copy()

    sentiment_results = result[text_column].apply(
        analyze_single_sentiment
    )

    sentiment_df = pd.DataFrame(
        sentiment_results.tolist(),
        index=result.index
    )

    result = pd.concat(
        [result, sentiment_df],
        axis=1
    )

    return result


# =========================================================
# 6. Summarize sentiment distribution
# =========================================================

def summarize_sentiment(result):

    if result is None or result.empty:
        raise ValueError("Input result is empty.")

    if "sentiment_label" not in result.columns:
        raise ValueError(
            "Sentiment analysis has not been performed."
        )

    counts = result["sentiment_label"].value_counts()

    total = len(result)

    summary = pd.DataFrame({
        "sentiment": counts.index,
        "feedback_count": counts.values,
        "percentage": [
            round(count / total * 100, 2)
            for count in counts.values
        ]
    })

    return summary.reset_index(drop=True)
