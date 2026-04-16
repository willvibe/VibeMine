import axios from 'axios';

const API_KEY_KEY = 'vibemine_gemini_key';
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

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

/** 前端直接调用 Gemini REST API */
export async function callGemini(prompt: string, apiKey?: string): Promise<string> {
  const key = apiKey || getApiKey();
  if (!key) {
    return '**未设置 API Key**\n\n请在设置中填写 Gemini API Key 后再使用 AI 功能';
  }
  const url = `${GEMINI_API_URL}?key=${key}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
    }),
  });
  if (!res.ok) {
    const errData = await res.json().catch(() => null);
    const errMsg = errData?.error?.message || res.statusText;
    if (res.status === 429) {
      return `**AI 分析失败 - 配额已用尽**\n\n您的 Gemini API Key 免费额度已用完。请：\n1. 前往 [Google AI Studio](https://aistudio.google.com/) 升级计费计划\n2. 或在设置中更换其他 API Key\n3. 免费用户每分钟有请求次数限制，请稍后重试`;
    }
    return `**AI 分析失败**\n\nHTTP ${res.status}: ${errMsg.slice(0, 150)}`;
  }
  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '**AI 返回空结果，请稍后重试**';
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

export async function getAiEvaluation(sessionId: string) {
  const res = await api.get(`/train/ai-evaluation/${sessionId}`, { headers: authHeaders() });
  return res.data;
}

export async function getAiMisclassified(sessionId: string) {
  const res = await api.get(`/train/ai-misclassified/${sessionId}`, { headers: authHeaders() });
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

export async function getUploadAIInsight(filename: string) {
  const res = await api.get(`/upload/ai-insight/${filename}`, { headers: authHeaders() });
  return res.data;
}
