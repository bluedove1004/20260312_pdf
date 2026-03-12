/** Document list component: table of uploaded PDFs with status badges. */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import type { DocumentListItem } from '../types';

interface DocumentListProps {
  documents: DocumentListItem[];
  onRefresh: () => void;
}

const STATUS_LABELS: Record<string, string> = {
  pending: '대기',
  processing: '처리 중',
  completed: '완료',
  failed: '실패',
};

const StatusBadge: React.FC<{ status: string }> = ({ status }) => (
  <span className={`status-badge status-${status}`}>
    {status === 'processing' && (
      <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
    )}
    {STATUS_LABELS[status] ?? status}
  </span>
);

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString('ko-KR', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

const DocumentList: React.FC<DocumentListProps> = ({ documents, onRefresh }) => {
  const navigate = useNavigate();

  if (documents.length === 0) {
    return (
      <div className="glass-card p-10 text-center text-gray-500 animate-fade-in">
        <svg className="w-12 h-12 mx-auto mb-3 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p className="text-sm">업로드된 문서가 없습니다</p>
        <p className="text-xs mt-1">위에서 PDF 파일을 업로드해 주세요</p>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden animate-slide-up">
      <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
        <h2 className="text-base font-semibold text-gray-200">문서 목록</h2>
        <button onClick={onRefresh} className="btn-ghost text-xs gap-1.5" id="refresh-documents-btn">
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
              d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          새로고침
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-gray-800">
              <th className="text-left px-5 py-3 font-medium">파일명</th>
              <th className="text-center px-4 py-3 font-medium">페이지</th>
              <th className="text-center px-4 py-3 font-medium">상태</th>
              <th className="text-right px-5 py-3 font-medium">생성일</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/60">
            {documents.map((doc) => (
              <tr
                key={doc.document_id}
                onClick={() => doc.status === 'completed' && navigate(`/documents/${doc.document_id}`)}
                className={`transition-colors duration-150
                  ${doc.status === 'completed' ? 'hover:bg-gray-800/50 cursor-pointer' : 'cursor-default'}`}
                id={`doc-row-${doc.document_id}`}
              >
                <td className="px-5 py-3.5">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gray-800 rounded-lg flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 1.5L18.5 9H13V3.5zM8 17h8v-1H8v1zm0-3h8v-1H8v1zm0-3h5v-1H8v1z" />
                      </svg>
                    </div>
                    <span className="text-gray-200 truncate max-w-[200px] font-medium">{doc.filename}</span>
                  </div>
                </td>
                <td className="px-4 py-3.5 text-center text-gray-400 tabular-nums">{doc.total_pages || '—'}</td>
                <td className="px-4 py-3.5 text-center"><StatusBadge status={doc.status} /></td>
                <td className="px-5 py-3.5 text-right text-gray-500 text-xs tabular-nums">{formatDate(doc.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DocumentList;
