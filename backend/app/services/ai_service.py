import json
from google import genai
from app.config import GEMINI_API_KEY

GEMINI_CLIENT = None

def get_client():
    global GEMINI_CLIENT
    if GEMINI_CLIENT is None:
        GEMINI_CLIENT = genai.Client(api_key=GEMINI_API_KEY, http_options={"timeout": 30})
    return GEMINI_CLIENT


def get_data_insight(profile: dict) -> str:
    try:
        client = get_client()

        col_summaries = []
        for col in profile["columns"]:
            line = f"- {col['name']} ({col['dtype']}): 缺失{col['missing_count']}个({col['missing_ratio']*100:.1f}%), 唯一值{col['unique_count']}个"
            if "mean" in col and col["mean"] is not None:
                line += f", 均值={col['mean']}, 标准差={col['std']}"
            if "top_value" in col and col["top_value"] is not None:
                line += f", 最频繁值={col['top_value']}"
            col_summaries.append(line)

        col_text = "\n".join(col_summaries)
        shape_text = f"{profile['shape'][0]}行 x {profile['shape'][1]}列"

        prompt = f"""你是一个资深的数据挖掘专家。用户刚上传了一份数据集，基本信息如下：
数据规模：{shape_text}
各列详情：
{col_text}

请简要分析该数据的潜在问题（如缺失值处理建议、数据分布），并根据特征判断这更适合做分类还是回归任务，推荐2种最合适的初阶算法。语言要专业且带有引导性，格式使用 Markdown，字数控制在 300 字以内。"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"**AI 分析暂时不可用**\n\n错误信息：{str(e)[:100]}"


def get_model_evaluation(metrics_table: list, target_column: str, task_type: str) -> str:
    try:
        client = get_client()

        metrics_text = json_to_markdown_table(metrics_table[:5])

        prompt = f"""你是一个资深的数据科学家。AutoML 引擎刚刚完成了模型对比，各项指标结果如下：

{metrics_text}

用户的目标列是：{target_column}
任务类型是：{task_type}

请为用户解读这份表格：哪一个模型综合表现最好？为什么？指出该模型在准确率和运算效率上的平衡点。如果有过拟合风险请提醒。语言直白易懂，格式使用 Markdown，字数控制在 400 字以内。"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"**AI 评估暂时不可用**\n\n错误信息：{str(e)[:100]}"


def json_to_markdown_table(data: list) -> str:
    if not data:
        return "无数据"
    headers = list(data[0].keys())
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    rows = []
    for row in data:
        row_line = "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
        rows.append(row_line)
    return "\n".join([header_line, separator] + rows)