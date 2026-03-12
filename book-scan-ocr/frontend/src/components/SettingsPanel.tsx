/** Settings panel: OCR provider, API key, preprocessing toggles, DPI. */

import React, { useEffect, useState } from 'react';
import { getSettings, saveSettings } from '../services/api';
import type { OCRSettings } from '../types';

const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<Partial<OCRSettings>>({
    ocr_provider: 'paddleocr',
    dpi: 300,
    preprocessing: { grayscale: true, binarization: false, denoise: false, deskew: false },
  });
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getSettings().then((s) => {
      setSettings(s);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      await saveSettings({ ...settings, api_key: apiKey || undefined });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } finally {
      setSaving(false);
    }
  };

  const updatePreprocessing = (key: string, val: boolean) => {
    setSettings((prev) => ({
      ...prev,
      preprocessing: { ...prev.preprocessing!, [key]: val } as OCRSettings['preprocessing'],
    }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-40">
        <div className="w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      <h1 className="text-2xl font-bold text-gray-100">설정</h1>

      {/* OCR Provider */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="text-base font-semibold text-gray-200">OCR 엔진</h2>
        <div className="grid grid-cols-3 gap-3">
          {(['paddleocr', 'google_vision', 'azure'] as const).map((provider) => (
            <button
              key={provider}
              id={`provider-${provider}`}
              onClick={() => setSettings((p) => ({ ...p, ocr_provider: provider }))}
              className={`py-3 px-4 rounded-xl border text-sm font-medium transition-all duration-200
                ${settings.ocr_provider === provider
                  ? 'border-brand-500 bg-brand-600/20 text-brand-300'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300'}`}
            >
              {provider === 'paddleocr' ? 'PaddleOCR' : provider === 'google_vision' ? 'Google Vision' : 'Azure OCR'}
            </button>
          ))}
        </div>
      </div>

      {/* API Key */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="text-base font-semibold text-gray-200">외부 API Key</h2>
        <p className="text-xs text-gray-500">PaddleOCR는 API Key가 필요 없습니다. Google Vision 또는 Azure 사용 시 입력하세요.</p>
        <div className="relative">
          <input
            id="api-key-input"
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="API Key를 입력하세요"
            className="input-field pr-12"
          />
          <button
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
            id="toggle-api-key-visibility"
            type="button"
          >
            {showKey
              ? <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" /></svg>
              : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
            }
          </button>
        </div>
      </div>

      {/* DPI */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="text-base font-semibold text-gray-200">PDF 변환 DPI</h2>
        <div className="flex gap-3">
          {([150, 200, 300, 400] as const).map((dpi) => (
            <button
              key={dpi}
              id={`dpi-${dpi}`}
              onClick={() => setSettings((p) => ({ ...p, dpi }))}
              className={`flex-1 py-2.5 rounded-xl border text-sm font-medium transition-all duration-200
                ${settings.dpi === dpi
                  ? 'border-brand-500 bg-brand-600/20 text-brand-300'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600'}`}
            >
              {dpi}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500">높은 DPI는 처리 시간이 길어지지만 인식률이 향상됩니다. 스캔 품질에 따라 300 권장.</p>
      </div>

      {/* Preprocessing */}
      <div className="glass-card p-6 space-y-4">
        <h2 className="text-base font-semibold text-gray-200">이미지 전처리</h2>
        <div className="grid grid-cols-2 gap-4">
          {[
            { key: 'grayscale', label: '그레이스케일 변환', desc: '컬러를 흑백으로 변환' },
            { key: 'binarization', label: '이진화 (Otsu)', desc: '경계를 선명하게 처리' },
            { key: 'denoise', label: '노이즈 제거', desc: '가우시안 블러로 잡음 제거' },
            { key: 'deskew', label: '기울기 보정', desc: '비뚤어진 페이지 교정' },
          ].map(({ key, label, desc }) => {
            const value = settings.preprocessing?.[key as keyof typeof settings.preprocessing] ?? false;
            return (
              <label
                key={key}
                className={`flex items-start gap-3 p-4 rounded-xl border cursor-pointer transition-all duration-200
                  ${value ? 'border-brand-500/50 bg-brand-950/20' : 'border-gray-700 hover:border-gray-600'}`}
                id={`toggle-${key}`}
              >
                <div className="flex-shrink-0 mt-0.5">
                  <div
                    onClick={() => updatePreprocessing(key, !value)}
                    className={`w-10 h-6 rounded-full transition-colors duration-300 relative cursor-pointer
                      ${value ? 'bg-brand-600' : 'bg-gray-700'}`}
                  >
                    <div className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform duration-300
                      ${value ? 'translate-x-5' : 'translate-x-1'}`} />
                  </div>
                </div>
                <div>
                  <p className="text-sm font-medium text-gray-200">{label}</p>
                  <p className="text-xs text-gray-500 mt-0.5">{desc}</p>
                </div>
              </label>
            );
          })}
        </div>
      </div>

      {/* Save Button */}
      <div className="flex items-center gap-4">
        <button
          onClick={handleSave}
          disabled={saving}
          className="btn-primary"
          id="save-settings-btn"
        >
          {saving ? (
            <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          )}
          {saving ? '저장 중…' : '설정 저장'}
        </button>
        {saved && (
          <span className="text-sm text-emerald-400 animate-fade-in flex items-center gap-1.5">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            저장되었습니다
          </span>
        )}
      </div>
    </div>
  );
};

export default SettingsPanel;
