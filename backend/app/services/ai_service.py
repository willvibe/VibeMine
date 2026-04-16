import json
import httpx
from google import genai
from app.config import GEMINI_API_KEY

_CLIENT_CACHE = {}


def _get_proxy() -> str | None:
    import os
    return os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")


def get_client(api_key: str = None) -> genai.Client:
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError("Gemini API Key 未设置，请在设置中配置")
    if key not in _CLIENT_CACHE:
        http_opts = {"timeout": 60}
        proxy = _get_proxy()
        if proxy:
            http_opts["proxy"] = proxy
        _CLIENT_CACHE[key] = genai.Client(api_key=key, http_options=http_opts)
    return _CLIENT_CACHE[key]


def get_data_insight(profile: dict, api_key: str = None) -> str:
    try:
        client = get_client(api_key)

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


def get_model_evaluation(metrics_table: list, target_column: str, task_type: str, api_key: str = None) -> str:
    try:
        client = get_client(api_key)

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


def get_misclassified_analysis(misclassified_samples: list, target_column: str, feature_importance: dict, api_key: str = None) -> str:
    if not misclassified_samples or len(misclassified_samples) == 0:
        return "**错误样本分析**\n\n本次训练中未出现预测错误的样本，说明模型在验证集上表现良好。"
    try:
        client = get_client(api_key)
        sample_count = len(misclassified_samples)
        display_samples = misclassified_samples[:20]

        feature_text = ""
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            feature_text = "Top 5 重要特征：\n" + "\n".join([f"- {k}: {v:.4f}" for k, v in top_features])

        sample_rows = []
        for s in display_samples:
            row_str = ", ".join([f"{k}={v}" for k, v in s.items() if k not in ('prediction_label', 'prediction_label_text')])
            sample_rows.append(f"- [{s.get(target_column, '?')}] 预测为 [{s.get('prediction_label', '?')}]: {row_str}")
        samples_text = "\n".join(sample_rows)

        prompt = f"""你是一个资深的数据科学家。用户正在进行二分类/多分类任务，模型在验证集上出现了一些预测错误的样本。

错误样本统计：共 {sample_count} 条（显示前 {len(display_samples)} 条）

Top 5 重要特征及权重：
{feature_text}

错误样本详情：
{samples_text}

请分析这些错误样本的特点：是否存在某类样本被集中误判？是否存在特征缺失或异常值导致的误判？并给出2-3条具体的优化建议（如：增加样本量、特征工程、调整阈值、换用其他算法等）。语言直白专业，格式使用 Markdown，控制在 300 字以内。"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return f"**错误样本分析**\n\n{response.text}"
    except Exception as e:
        return f"**错误样本分析暂时不可用**\n\n错误信息：{str(e)[:100]}"


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
