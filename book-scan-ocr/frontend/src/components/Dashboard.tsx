/**
 * Dashboard: main landing page combining file uploader, progress tracking,
 * and the list of all uploaded documents.
 */

import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listDocuments } from '../services/api';
import type { DocumentListItem, DocumentStatusResponse } from '../types';
import DocumentList from './DocumentList';
import FileUploader from './FileUploader';
import ProgressBar from './ProgressBar';
import { useDocumentPolling } from '../hooks/useDocumentPolling';

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<DocumentListItem[]>([]);
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [processingFilename, setProcessingFilename] = useState<string>('');

  const { status: pollStatus } = useDocumentPolling({
    documentId: processingId,
    enabled: processingId !== null,
    onComplete: (s: DocumentStatusResponse) => {
      if (s.status === 'completed') {
        refreshDocuments();
        setProcessingId(null);
        navigate(`/documents/${s.document_id}`);
      } else {
        setProcessingId(null);
        refreshDocuments();
      }
    },
  });

  const refreshDocuments = useCallback(() => {
    listDocuments().then(setDocuments).catch(console.error);
  }, []);

  useEffect(() => { refreshDocuments(); }, [refreshDocuments]);

  const handleUploaded = (documentId: string, filename: string) => {
    setProcessingId(documentId);
    setProcessingFilename(filename);
    refreshDocuments();
  };

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 space-y-8 animate-fade-in">
      {/* Hero */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 bg-brand-900/40 border border-brand-800/50 rounded-full text-xs text-brand-300 mb-4">
          <span className="w-1.5 h-1.5 bg-brand-400 rounded-full animate-pulse" />
          PaddleOCR 한국어 지원
        </div>
        <h1 className="text-4xl font-bold text-gray-100 tracking-tight">
          Book Scan <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-purple-400">OCR</span>
        </h1>
        <p className="text-gray-400 mt-2 text-base">스캔된 책 PDF를 업로드하면 AI가 텍스트를 추출합니다</p>
      </div>

      {/* Upload */}
      <div className="glass-card p-6">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">PDF 업로드</h2>
        <FileUploader onUploaded={handleUploaded} />
      </div>

      {/* Progress (while processing) */}
      {processingId && pollStatus && (
        <ProgressBar status={pollStatus} filename={processingFilename} />
      )}

      {/* Document list */}
      <DocumentList documents={documents} onRefresh={refreshDocuments} />
    </div>
  );
};

export default Dashboard;
