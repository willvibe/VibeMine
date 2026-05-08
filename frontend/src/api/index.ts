import axios from 'axios';

const API_KEY_KEY = 'vibemine_gemini_key';

export function getApiKey(): string | null {
  return localStorage.getItem(API_KEY_KEY);
}

const BASE_URL = import.meta.env.PROD ? 'http://45.113.2.167:8000/api' : '/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 600000,
});

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { 'x-api-key': key } : {};
}

/** 通过后端代理调用 Gemini API（支持代理配置） */
export async function callGemini(prompt: string, apiKey?: string): Promise<string> {
  const key = apiKey || getApiKey();
  if (!key) {
    return '**未设置 API Key**\n\n请在设置中填写 Gemini API Key 后再使用 AI 功能';
  }
  try {
    const res = await api.post('/ai/gemini', { prompt, api_key: key }, { headers: authHeaders() });
    return res.data.result || '**AI 返回空结果，请稍后重试**';
  } catch (err: any) {
    const detail = err?.response?.data?.detail || err.message || String(err);
    if (err?.response?.status === 429) {
      return `**AI 分析失败 - 配额已用尽**\n\n您的 Gemini API Key 免费额度已用完。请：\n1. 前往 [Google AI Studio](https://aistudio.google.com/) 升级计费计划\n2. 或在设置中更换其他 API Key\n3. 免费用户每分钟有请求次数限制，请稍后重试`;
    }
    return `**AI 分析失败**\n\n${detail.slice(0, 200)}`;
  }
}

export async function uploadFile(formData: FormData) {
  const res = await api.post('/upload', formData, {
    headers: { ...authHeaders() },
  });
  return res.data;
}

export async function startTraining(
  params: {
    filename: string;
    task_type: string;
    target_column: string;
    selected_models?: string[];
    ignore_columns?: string[];
    use_smote?: boolean;
    use_outlier_removal?: boolean;
    use_advanced_imputation?: boolean;
    use_stratified_cv?: boolean;
    use_tuning?: boolean;
    use_ensembling?: boolean;
  },
) {
  const res = await api.post('/train', params, { headers: authHeaders() });
  return res.data;
}

export async function getTrainStatus(sessionId: string) {
  const res = await api.get(`/train/status/${sessionId}`, { headers: authHeaders() });
  return res.data;
}

export async function stopTraining(sessionId: string) {
  const res = await api.post(`/train/stop/${sessionId}`, null, { headers: authHeaders() });
  return res.data;
}

export async function downloadModel(modelId: string) {
  const downloadUrl = `${BASE_URL}/download/${modelId}`;
  const headers = authHeaders();
  const response = await axios.get(downloadUrl, {
    headers,
    responseType: 'blob',
  });
  return response.data;
}
