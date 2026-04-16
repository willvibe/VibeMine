import axios from 'axios';

const API_KEY_KEY = 'vibemine_gemini_key';

function getApiKey(): string | null {
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
