# InsightFlow

### AI-Powered Consumer Insight & Decision Intelligence Platform

**From Consumer Feedback to Actionable Insights.**

InsightFlow 是一个面向消费者反馈分析的 AI 辅助洞察平台，旨在将非结构化的消费者评价转化为结构化数据、产品体验问题、消费者需求及可供业务参考的优化建议。

项目结合 Python 数据处理、文本分析、情感分析与大语言模型能力，探索 AI 在消费者研究、产品优化和营销决策中的应用。

---

## 01. Project Overview

消费者评价中不仅包含对产品的直接满意或不满意，也可能隐藏着具体使用问题、未被满足的需求以及潜在的产品改进方向。

InsightFlow 尝试建立一套从原始反馈到业务洞察的分析流程，帮助使用者更系统地理解消费者反馈，而不只是统计评分或情感倾向。

项目重点关注三个问题：

- 消费者主要在讨论哪些产品体验？
- 反馈中反复出现哪些问题与需求？
- 如何将零散的消费者意见转化为后续产品优化的参考？

## 02. Core Features

| Module | Key Capabilities |
|---|---|
| Data Processing | CSV 数据读取、清洗与质量检查 |
| Data Quality Analysis | 异常数据识别与数据质量分析 |
| Sentiment Analysis | 消费者情感倾向识别与分布分析 |
| Text Analysis | 反馈文本处理与主题识别 |
| AI Feedback Analysis | 正面评价、负面评价及混合情感分析 |
| Pain Point Mining | 产品使用痛点识别 |
| Consumer Needs | 显性需求与隐含需求提取 |
| Product Comparison | 不同产品的评分及情感表现对比 |
| Business Insights | 产品优化及业务改进建议 |
| Interactive Dashboard | 基于 Streamlit 的交互式分析界面 |

## 03. AI-Powered Consumer Insight

项目使用 DeepSeek API 辅助分析消费者原始反馈。

AI 分析主要包括：

- **Sentiment**：整体情感倾向识别，包括正面、负面及混合情感。
- **Positive Aspects**：消费者认可的产品特点。
- **Negative Aspects**：消费者表达的不满。
- **Pain Points**：使用过程中遇到的具体问题。
- **Explicit Needs**：消费者直接表达的需求。
- **Inferred Needs**：结合反馈语境识别的潜在需求。
- **Suggestions**：针对反馈内容提出的改进建议。
- **Evidence**：支持分析结果的原始文本依据。

项目特别关注显性需求与隐含需求之间的区别，尝试从消费者的具体表达中提取更深层次的产品体验信息。

## 04. Analytical Workflow

```text
Raw Consumer Feedback
          |
          v
     Data Loading
          |
          v
 Data Cleaning & Quality Control
          |
          v
     Text Analysis
          |
          v
   Sentiment Analysis
          |
          v
  AI-Powered Feedback Analysis
          |
          v
 Pain Points & Consumer Needs
          |
          v
 Product Performance Comparison
          |
          v
 Business Insights & Suggestions
```

## 05. Visualization & Results

以下展示项目生成的部分数据分析图表。

### Data Quality Analysis

![Data Quality Distribution](screenshots/data_quality_distribution.png)

### Consumer Sentiment Analysis

![Sentiment Distribution](screenshots/sentiment_distribution.png)

### Product Sentiment Comparison

![Product Sentiment Comparison](screenshots/product_sentiment_comparison.png)

### Feedback Trend Analysis

![Feedback Trend](screenshots/feedback_trend.png)

## 06. Technology Stack

| Technology | Application |
|---|---|
| Python | Core development |
| Pandas | Data processing and analysis |
| Scikit-learn | Text analysis and classification support |
| Plotly | Interactive data visualization |
| Streamlit | Interactive dashboard |
| DeepSeek API | AI-powered feedback analysis |
| OpenAI Python SDK | API-compatible model access |
| python-dotenv | Environment variable management |

## 07. Project Structure

```text
InsightFlow/
├── dashboard.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── raw/
│       ├── sports_feedback.csv
│       └── sample_feedback.csv
│
├── screenshots/
│   ├── data_quality_distribution.png
│   ├── feedback_trend.png
│   ├── product_sentiment_comparison.png
│   └── sentiment_distribution.png
│
└── src/
    ├── data_loader.py
    ├── data_cleaner.py
    ├── text_analyzer.py
    ├── sentiment_analyzer.py
    ├── llm_analyzer.py
    ├── insight_generator.py
    ├── business_insight_generator.py
    └── visualizer.py
```

## 08. Installation & Usage

### Step 1. Clone the repository

```bash
git clone https://github.com/ruixi-lin/InsightFlow.git
cd InsightFlow
```

### Step 2. Install dependencies

建议使用 Python 3.11 或 3.12 创建独立虚拟环境。

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

### Step 3. Configure API Key

本地运行时，在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=your_api_key_here
```

部署到 Streamlit Community Cloud 时，在应用的 **Settings → Secrets** 中添加：

```toml
DEEPSEEK_API_KEY = "your_api_key_here"
```

请勿将真实 API Key 上传至 GitHub。未配置 API Key 时，平台仍可使用示例数据体验常规分析模块，AI 深度分析功能将不可用。

### Step 4. Run the application

```bash
streamlit run dashboard.py
```

## 09. Online Demo

**Live Demo:** Coming Soon

在线体验版本正在部署中，正式上线后将在此提供公开访问链接。

## 10. Project Notes

- 项目以运动消费产品反馈作为示例分析场景。
- 示例数据用于功能演示，不代表真实市场调查结果。
- AI 分析结果属于辅助性洞察，需要结合原始反馈进行人工核查。
- 在线 AI 分析功能需要配置有效的 DeepSeek API Key。

---

**InsightFlow V1.3**

AI-Powered Consumer Insight & Decision Intelligence

Developed by Ruixi Lin
