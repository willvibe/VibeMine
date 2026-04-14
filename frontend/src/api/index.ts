import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.PROD ? 'http://45.113.2.167:8000/api' : '/api',
  timeout: 600000,
});

export async function uploadFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
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
  },
) {
  const res = await api.post('/train', params);
  return res.data;
}

export async function getTrainStatus(sessionId: string) {
  const res = await api.get(`/train/status/${sessionId}`);
  return res.data;
}

export async function stopTraining(sessionId: string) {
  const res = await api.post(`/train/stop/${sessionId}`);
  return res.data;
}

export async function downloadModel(modelId: string) {
  const res = await api.get(`/download/${modelId}`, {
    responseType: 'blob',
  });
  return res.data;
}