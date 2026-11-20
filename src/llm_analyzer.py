
import os
import json
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables
load_dotenv()

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def is_llm_configured():
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def get_llm_config():
    return {
        "provider": "DeepSeek",
        "model": DEFAULT_MODEL,
        "base_url": DEFAULT_BASE_URL,
        "configured": is_llm_configured(),
    }


def get_llm_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("DeepSeek API Key is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url=DEFAULT_BASE_URL,
        timeout=30.0,
        max_retries=1,
    )


def test_llm_connection():
    """Send a minimal request to verify API connectivity."""

    client = get_llm_client()

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Reply briefly in Chinese."
            },
            {
                "role": "user",
                "content": "请回复：InsightFlow连接成功。"
            }
        ],
        max_tokens=30,
        extra_body={
            "thinking": {"type": "disabled"}
        }
    )

    return {
        "status": "success",
        "model": response.model,
        "reply": response.choices[0].message.content,
        "total_tokens": response.usage.total_tokens,
    }

def analyze_with_llm(feedback_text):
    """
    Analyze one consumer feedback record using DeepSeek.
    Return structured consumer insights.
    """

    if not feedback_text or not str(feedback_text).strip():
        raise ValueError("Feedback text cannot be empty.")

    client = get_llm_client()

    
    system_prompt = """
You are a professional consumer insight analyst.

Your task is to identify consumer sentiment, product
evaluation, pain points, explicit needs, inferred needs,
and relevant suggestions from consumer feedback.

Analyze the complete meaning and context of the original
feedback. Do not rely on isolated keywords.

Return valid JSON only.

The JSON must contain exactly these fields:

{
    "sentiment": "正面/负面/中性/混合",
    "positive_aspects": [],
    "negative_aspects": [],
    "pain_points": [],
    "explicit_needs": [],
    "inferred_needs": [],
    "suggestions": [],
    "evidence": [],
    "confidence": 0.0
}

I. SENTIMENT CLASSIFICATION

Use exactly one of the following labels:

1. Positive:
   The consumer explicitly expresses satisfaction,
   approval, or a favorable experience.

2. Negative:
   The consumer expresses dissatisfaction,
   disappointment, frustration, or an unfavorable
   experience.

3. Neutral:
   The feedback contains factual descriptions,
   ambiguous opinions, or product requests without
   a sufficiently clear positive or negative evaluation.

4. Mixed:
   The feedback contains both positive and negative
   evaluations, or follows an explicitly established
   mixed-sentiment annotation convention.

Additional rules:

- Do not classify a product request alone as positive.
- Do not automatically classify every complaint
  as negative if the complete feedback is mixed.
- Interpret contrastive expressions such as
  "虽然...但是..." using the complete sentence.
- Interpret double negation according to its actual meaning.
- Do not automatically treat the absence of a defect
  as positive satisfaction.
- Mild dissatisfaction should not be exaggerated.
- When an opinion is ambiguous and no clear evaluation
  can be established, prefer Neutral.

Additional sentiment decision rules:

- The absence of a specific defect does not constitute
  explicit positive satisfaction.

- Mild expressions such as "还可以" should not
  automatically be treated as positive sentiment.
  When accompanied by explicit dissatisfaction,
  classify the overall feedback as Negative if no
  clear positive satisfaction is expressed.

- Mixed sentiment requires clearly expressed favorable
  and unfavorable evaluations. A weak or qualified
  expression alone does not establish positive sentiment.

- When mild approval such as "整体还可以" is
  followed by a specific criticism, do not automatically
  classify the feedback as Mixed.

- If the mild approval does not express clear satisfaction
  and the criticism is the only specific evaluation,
  classify the overall sentiment as Negative.

- For example:
  "整体还可以，就是包装稍微有点普通"
  should be classified as Negative rather than Mixed.

- Expressions such as "不能说完全不好，但也谈不上
  有多满意" should be classified as Neutral when
  neither clear satisfaction nor a specific problem
  is established.

II. POSITIVE AND NEGATIVE ASPECTS

positive_aspects:
List favorable product attributes or experiences
explicitly supported by the feedback.

negative_aspects:
List unfavorable product attributes or experiences
explicitly supported by the feedback.

Do not invent positive or negative evaluations.

III. PAIN POINT IDENTIFICATION

Pain points describe specific problems, limitations,
dissatisfaction, or unmet expectations experienced
by the consumer.

Rules:

- Extract only problems supported by the original text.
- Do not convert every negative word into a separate
  pain point.
- Do not include explicitly negated problems as actual
  defects.
- Combine duplicate descriptions of the same problem.
- Keep genuinely different problems separate.
- Describe the actual problem rather than inventing
- Do not create a separate generic pain point such as
  "poor overall experience" or "experience below expectations"
  when a specific functional failure already explains
  the consumer's dissatisfaction.

- Strong emotional expressions such as "very disappointing"
  or "terrible" do not constitute additional pain points
  when they merely emphasize an already identified problem.
  its technical cause.
- Do not confuse a pain point with a proposed solution.

- When multiple descriptions refer to the same underlying
  dissatisfaction, combine them into one concise pain point.
  Do not create separate pain points merely because the
  same issue is described from different perspectives.

- For example:
  "没有想象中那么好，用起来一般"
  => "实际使用体验未达到预期"
  Do not additionally output "使用体验不够好" as
  a separate pain point.

- When the feedback describes an inconvenience caused by
  carrying an accessory, identify the directly expressed
  inconvenience separately from its possible technical cause.

- For example:
  "每次出门都要带充电宝，实在有点麻烦"
  => "外出时需要携带充电宝"
  Do not automatically assert that the product's battery
  life is insufficient unless the feedback explicitly
  establishes this problem.

Additional pain point identification rules:

- Identify all distinct problems explicitly described
  in the feedback, even when multiple problems appear
  in the same sentence.

- Include both product-related problems and service-related
  problems, such as product durability issues and
  unresponsive customer service, when supported by
  the original feedback.

- When a specific failure event is described, identify
  the underlying problem without inventing its cause.

- For example:
  "用了三天就坏了，客服也一直不回复"
  => "产品使用寿命短"
  => "客服未及时回复"

- Merge semantically equivalent descriptions of the
  same problem. Do not report the same issue multiple
  times using different wording.

- For example, "电池续航能力不足" and
  "电池续航时间短" should be treated as one
  pain point when they refer to the same issue.

IV. EXPLICIT NEEDS

Explicit needs are requirements or usage goals directly
expressed by the consumer.

They include:

- Directly requested product functions.
- Explicitly requested improvements.
- Clearly stated usage goals or desired outcomes.

Examples:

"希望增加续航时间"
=> Increase battery life.

"最好还能支持快速充电"
=> Support fast charging.

"这样出门使用会方便很多"
=> Improve convenience during outdoor use.

Do not classify a complaint as an explicit need merely
because a possible solution can be imagined.

Additional explicit needs rules:

- Extract all distinct requirements and desired outcomes
  directly expressed in the original feedback.

- When multiple explicit requirements appear in one
  sentence, identify each requirement separately.

- Clearly stated usage goals and practical benefits
  should be retained as explicit needs when they are
  directly expressed by the consumer.

- For example:
  "希望增加续航时间，最好还能支持快速充电，
  这样出门使用会方便很多"
  => "增加续航时间"
  => "支持快速充电"
  => "提升外出使用的便利性"

- For example:
  "建议增加一个电量提醒功能，方便及时充电"
  => "增加电量提醒功能"
  => "方便及时充电"

- Do not generate an additional inferred need when
  the same underlying requirement or desired outcome
  has already been explicitly expressed.

V. INFERRED NEEDS

Inferred needs are reasonable underlying requirements
that are not directly stated as explicit requests,
but can be derived from the consumer's problem,
experience, or stated functional purpose.

Examples:

"每次出门都要带充电宝，实在有点麻烦"
=> Improve device battery life.

Rules:

- Inferred needs must have a clear basis in the original text.
- Do not invent motivations or expectations.
- Do not repeat explicit needs using different wording.
- Do not produce multiple redundant inferred needs
  describing essentially the same objective.
- Keep inferred needs at the level of the underlying
  consumer requirement, not an unsupported technical
  implementation.

Additional inferred needs rules:


0. When feedback expresses mild dissatisfaction with a
   specific product attribute, design element, or
   presentation aspect without explicitly requesting
   an improvement, infer a reasonable need to improve
   that specific aspect.

   For example:
   "整体还可以，就是包装稍微有点普通"
   => "改善包装设计或呈现效果"

   Do not leave inferred_needs empty merely because
   the complaint is mild or does not contain an
   explicit improvement request.

   Keep the inferred need specific to the aspect
   criticized. Do not generalize it into an overall
   experience improvement unless the feedback supports
   that broader interpretation.

1. When the consumer explicitly states a product
   requirement or desired outcome, classify it as
   an explicit need rather than generating a redundant
   inferred need.

2. Do not generate inferred needs when the feedback
   already expresses all relevant requirements explicitly.

3. When a complaint implies an underlying consumer
   requirement, identify the requirement at an appropriate
   level of abstraction without adding unnecessary details.



4. For example:
   "每次出门都要带充电宝，实在有点麻烦"
   => "提升设备续航能力"

   Do not add unsupported details about charging
   conditions or external charging scenarios.


5. When a consumer explicitly states both a requested
   product function and its intended usage goal, do not
   generate an additional inferred need that merely
   restates, paraphrases, or further abstracts the same
   function or usage goal.

   For example:
   "建议增加一个电量提醒功能，方便及时充电"
   => Explicit needs:
      "增加电量提醒功能"
      "方便及时充电"
   => Inferred needs:
      []

   Do not additionally generate:
   "帮助用户及时掌握电量状态"
   because this is a further abstraction of the
   explicitly stated usage goal, not a distinct need.

6. Before generating any inferred need, check whether
   its underlying objective is already covered by the
   explicit needs.

   If the explicit needs already capture the relevant
   product requirement and intended usage goal, return
   an empty inferred_needs list unless the original
   feedback provides evidence of another distinct,
   unmet requirement.

   Do not invent intermediate requirements merely
   because they could theoretically help achieve an
   explicitly stated goal.


7. When a specific functional failure fully explains
   the consumer's dissatisfaction, prioritize the
   corresponding functional recovery need.

   Do not generate an additional generic inferred need
   such as "improve the overall user experience" unless
   the feedback provides evidence of a separate,
   distinct experience-related requirement.

   For example:
   "太失望了，完全无法正常使用，真的非常糟糕"
   => "修复产品故障，恢复正常使用"

   Do not additionally generate:
   "改善整体使用体验".



VI. SUGGESTIONS

Suggestions are actionable improvement directions.

They should be relevant to identified pain points
or explicitly expressed needs.

Do not introduce unsupported technical solutions,
components, materials, or product modifications.

VII. EVIDENCE

Evidence must be copied verbatim from the original text.

Rules:

- Every evidence item must appear exactly in the
  original feedback.
- Do not paraphrase evidence.
- Do not fabricate quotations.
- Select concise evidence that directly supports
  the corresponding analysis.

VIII. OUTPUT QUALITY CONTROL

Before returning the JSON, check that:

1. All required fields are present.
2. Sentiment uses one of the four permitted labels.
3. All list fields contain strings only.
4. Pain points are grounded in the original text.
5. Explicit and inferred needs are distinguished correctly.
6. The same requirement is not unnecessarily repeated
   across multiple categories.
7. Suggestions do not introduce unsupported assumptions.
8. Evidence is copied verbatim.
9. Confidence is a number between 0 and 1.

Use concise, specific, and consistent Chinese expressions
for all analysis list items.

When evidence is insufficient, return an empty list
instead of making unsupported assumptions.

Lower confidence when the feedback is ambiguous.

Return valid JSON only.
Do not include Markdown code fences or explanatory text.
"""

    user_prompt = f"""
Please analyze the following consumer feedback:

{str(feedback_text).strip()}
"""

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=1200,
        extra_body={
            "thinking": {"type": "disabled"}
        }
    )

        # Validate API response
    if response is None:
        raise ValueError("LLM returned an empty response.")

    choices = getattr(response, "choices", None)

    if not choices:
        raise ValueError("LLM response contains no choices.")

    message = getattr(choices[0], "message", None)

    if message is None:
        raise ValueError("LLM response contains no message.")

    content = getattr(message, "content", None)

    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or invalid content.")

    # Parse JSON
    try:
        analysis = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        raise ValueError(
            f"LLM returned invalid JSON: {content}"
        )

    # Validate JSON root type
    if not isinstance(analysis, dict):
        raise ValueError("LLM response must be a JSON object.")

    # Validate required fields
    required_fields = [
        "sentiment",
        "positive_aspects",
        "negative_aspects",
        "pain_points",
        "explicit_needs",
        "inferred_needs",
        "suggestions",
        "evidence",
        "confidence",
    ]

    missing_fields = [
        field for field in required_fields
        if field not in analysis
    ]

    if missing_fields:
        raise ValueError(
            f"Missing fields in LLM response: {missing_fields}"
        )

    # Validate sentiment
    valid_sentiments = [
        "正面", "负面", "中性", "混合"
    ]

    if analysis["sentiment"] not in valid_sentiments:
        raise ValueError("Invalid sentiment label.")

    # Validate list fields
    list_fields = [
        "positive_aspects",
        "negative_aspects",
        "pain_points",
        "explicit_needs",
        "inferred_needs",
        "suggestions",
        "evidence",
    ]

    for field in list_fields:

        if not isinstance(analysis[field], list):
            raise ValueError(
                f"{field} must be a list."
            )

        if not all(
            isinstance(item, str)
            for item in analysis[field]
        ):
            raise ValueError(
                f"All items in {field} must be strings."
            )

    # Validate confidence
    confidence = analysis["confidence"]

    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
    ):
        raise ValueError("Confidence must be numeric.")

    if not 0 <= confidence <= 1:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    return {
        "status": "success",
        "model": response.model,
        "analysis": analysis,
        "total_tokens": response.usage.total_tokens,
    }

