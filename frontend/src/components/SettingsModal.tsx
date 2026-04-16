import { useState, useEffect } from 'react';
import { useI18n, API_KEY_KEY } from '../i18n/index';

interface SettingsModalProps {
  onClose: () => void;
}

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const { t, lang, setLang } = useI18n();
  const [apiKey, setApiKey] = useState('');
  const [proxy, setProxy] = useState('');
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem(API_KEY_KEY);
    if (stored) setApiKey(stored);
    const storedProxy = localStorage.getItem('GEMINI_PROXY');
    if (storedProxy) setProxy(storedProxy);
  }, []);

  const handleSave = () => {
    localStorage.setItem(API_KEY_KEY, apiKey.trim());
    localStorage.setItem('GEMINI_PROXY', proxy.trim());
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={onClose} />
      <div className="relative bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900">{t('settings')}</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-all"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-6 py-5 space-y-6">
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">{t('apiKey')}</label>
            <div className="relative">
              <input
                type="password"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                placeholder={t('apiKeyPlaceholder')}
                className="w-full px-3 py-2 pr-10 rounded-xl border border-gray-200 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 transition-all"
              />
              {apiKey && (
                <button
                  onClick={() => { navigator.clipboard.writeText(apiKey); setSaved(true); setTimeout(() => setSaved(false), 1500); }}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-indigo-500 transition-colors"
                  title="Copy"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </button>
              )}
            </div>
            <p className="text-xs text-gray-400 mt-1.5">{t('apiKeyTip')}</p>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Gemini 代理地址</label>
            <input
              type="text"
              value={proxy}
              onChange={e => setProxy(e.target.value)}
              placeholder="例如: http://127.0.0.1:7890"
              className="w-full px-3 py-2 rounded-xl border border-gray-200 text-sm text-gray-800 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-400 transition-all"
            />
            <p className="text-xs text-gray-400 mt-1.5">国内服务器需配置代理才能访问 Gemini API</p>
          </div>

        </div>

        <div className="flex items-center justify-end gap-3 px-6 py-4 border-t border-gray-100 bg-gray-50/50">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 transition-all"
          >
            {t('cancel')}
          </button>
          <button
            onClick={handleSave}
            className="px-5 py-2 rounded-xl text-sm font-semibold bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-sm hover:shadow-md transition-all"
          >
            {saved ? t('saved') : t('save')}
          </button>
        </div>
      </div>
    </div>
  );
}
