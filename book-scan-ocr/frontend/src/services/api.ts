/**
 * Centralised API client.
 * All HTTP calls go through this module.
 * Vite proxies /api → http://localhost:8000
 */

import axios from 'axios';
import type {
  DocumentListItem,
  DocumentResult,
  DocumentStatusResponse,
  OCRSettings,
  PageResult,
  UploadResponse,
} from '../types';

const api = axios.create({ baseURL: '/api' });

// ──────────────────────────────────────────────
// Documents
// ──────────────────────────────────────────────

export async function uploadDocument(
  file: File,
  onProgress?: (pct: number) => void
): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded * 100) / e.total));
      }
    },
  });
  return data;
}

export async function listDocuments(): Promise<DocumentListItem[]> {
  const { data } = await api.get<DocumentListItem[]>('/documents');
  return data;
}

export async function getDocumentStatus(id: string): Promise<DocumentStatusResponse> {
  const { data } = await api.get<DocumentStatusResponse>(`/documents/${id}/status`);
  return data;
}

export async function getDocument(id: string): Promise<DocumentResult> {
  const { data } = await api.get<DocumentResult>(`/documents/${id}`);
  return data;
}

export async function getPage(id: string, page: number): Promise<PageResult> {
  const { data } = await api.get<PageResult>(`/documents/${id}/pages/${page}`);
  return data;
}

export function getPageImageUrl(id: string, page: number): string {
  return `/api/documents/${id}/pages/${page}/image`;
}

export function getDownloadUrl(id: string): string {
  return `/api/documents/${id}/download`;
}

// ──────────────────────────────────────────────
// Settings
// ──────────────────────────────────────────────

export async function getSettings(): Promise<OCRSettings> {
  const { data } = await api.get<OCRSettings>('/settings');
  return data;
}

export async function saveSettings(settings: Partial<OCRSettings> & { api_key?: string }): Promise<void> {
  await api.post('/settings', settings);
}
