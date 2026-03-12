/**
 * DocumentViewer: 3-panel layout — page list, scanned image, OCR result tabs.
 * Handles page selection, image zoom, and tab switching.
 */

import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getDocument, getDownloadUrl, getPage, getPageImageUrl } from '../services/api';
import type { DocumentResult, PageResult } from '../types';
import BlockView from './BlockView';
import JsonView from './JsonView';
import PageNavigation from './PageNavigation';
import TextView from './TextView';
import { usePageNavigation } from '../hooks/usePageNavigation';

type Tab = 'text' | 'json' | 'blocks';

const DocumentViewer: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [document, setDocument] = useState<DocumentResult | null>(null);
  const [currentPageData, setCurrentPageData] = useState<PageResult | null>(null);
  const [loadingDoc, setLoadingDoc] = useState(true);
  const [loadingPage, setLoadingPage] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>('text');
  const [zoom, setZoom] = useState(1);
  const [imgError, setImgError] = useState(false);

  const nav = usePageNavigation({ totalPages: document?.total_pages ?? 1 });

  // Load document metadata
  useEffect(() => {
    if (!id) return;
    getDocument(id).then((doc) => {
      setDocument(doc);
      setLoadingDoc(false);
    }).catch(() => setLoadingDoc(false));
  }, [id]);

  // Load page data when page changes
  useEffect(() => {
    if (!id || !document) return;
    setLoadingPage(true);
    setImgError(false);
    getPage(id, nav.currentPage).then((p) => {
      setCurrentPageData(p);
    }).finally(() => setLoadingPage(false));
  }, [id, document, nav.currentPage]);

  if (loadingDoc) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  if (!document) {
    return (
      <div className="text-center py-16 text-gray-500">
        <p className="text-lg">문서를 찾을 수 없습니다</p>
        <Link to="/" className="btn-secondary mt-4 inline-flex">← 목록으로</Link>
      </div>
    );
  }

  const imageUrl = id ? getPageImageUrl(id, nav.currentPage) : '';

  return (
    <div className="flex flex-col h-full animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/60 backdrop-blur">
        <div className="flex items-center gap-4">
          <Link to="/" className="btn-ghost text-sm" id="back-to-dashboard">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            목록
          </Link>
          <div className="h-4 w-px bg-gray-700" />
          <div>
            <h1 className="text-sm font-semibold text-gray-200 truncate max-w-sm">{document.filename}</h1>
            <p className="text-xs text-gray-500">{document.total_pages}페이지 · {document.ocr_engine}</p>
          </div>
        </div>
        <a
          href={id ? getDownloadUrl(id) : '#'}
          download
          className="btn-secondary text-xs"
          id="download-json-btn"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
          </svg>
          JSON 다운로드
        </a>
      </div>

      {/* 3-panel layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: page list sidebar */}
        <div className="w-20 flex-shrink-0 border-r border-gray-800 bg-gray-900/40 overflow-y-auto">
          {Array.from({ length: document.total_pages }, (_, i) => i + 1).map((pg) => (
            <button
              key={pg}
              id={`page-thumb-${pg}`}
              onClick={() => nav.goToPage(pg)}
              className={`w-full py-3 flex flex-col items-center gap-1 text-xs transition-colors duration-150
                ${nav.currentPage === pg
                  ? 'bg-brand-600/30 text-brand-300 border-r-2 border-brand-500'
                  : 'text-gray-500 hover:bg-gray-800 hover:text-gray-300'}`}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              {pg}
            </button>
          ))}
        </div>

        {/* Center: scanned image */}
        <div className="flex-1 flex flex-col border-r border-gray-800 bg-gray-950/50 min-w-0">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-900/40">
            <span className="text-xs text-gray-500">원본 이미지</span>
            <div className="flex items-center gap-2">
              <button onClick={() => setZoom((z) => Math.max(0.25, z - 0.25))} className="btn-ghost text-xs px-2 py-1" id="zoom-out-btn">−</button>
              <span className="text-xs text-gray-400 tabular-nums w-12 text-center">{Math.round(zoom * 100)}%</span>
              <button onClick={() => setZoom((z) => Math.min(3, z + 0.25))} className="btn-ghost text-xs px-2 py-1" id="zoom-in-btn">+</button>
              <button onClick={() => setZoom(1)} className="btn-ghost text-xs px-2 py-1" id="zoom-reset-btn">리셋</button>
            </div>
          </div>
          <div className="flex-1 overflow-auto flex items-start justify-center p-4">
            {imgError ? (
              <div className="text-gray-600 text-sm text-center mt-20">
                <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                이미지를 불러올 수 없습니다
              </div>
            ) : (
              <img
                src={imageUrl}
                alt={`페이지 ${nav.currentPage}`}
                id={`scanned-image-p${nav.currentPage}`}
                className="rounded-lg shadow-2xl object-contain transition-transform duration-200"
                style={{ transform: `scale(${zoom})`, transformOrigin: 'top center', maxWidth: 'none' }}
                onError={() => setImgError(true)}
              />
            )}
          </div>
          <PageNavigation
            currentPage={nav.currentPage}
            totalPages={document.total_pages}
            canGoPrev={nav.canGoPrev}
            canGoNext={nav.canGoNext}
            onGoPrev={nav.goPrev}
            onGoNext={nav.goNext}
            onGoToPage={nav.goToPage}
          />
        </div>

        {/* Right: OCR result tabs */}
        <div className="w-96 flex-shrink-0 flex flex-col bg-gray-900/30">
          {/* Tab bar */}
          <div className="flex gap-1 p-3 border-b border-gray-800 bg-gray-900/40">
            {(['text', 'json', 'blocks'] as Tab[]).map((tab) => (
              <button
                key={tab}
                id={`tab-${tab}`}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all duration-200
                  ${activeTab === tab ? 'tab-active' : 'tab-inactive'}`}
              >
                {tab === 'text' ? '텍스트' : tab === 'json' ? 'JSON' : '블록'}
              </button>
            ))}
          </div>

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto p-4">
            {loadingPage ? (
              <div className="flex items-center justify-center h-32">
                <div className="w-6 h-6 border-2 border-brand-600 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : currentPageData ? (
              <>
                {activeTab === 'text' && <TextView page={currentPageData} />}
                {activeTab === 'json' && <JsonView page={currentPageData} />}
                {activeTab === 'blocks' && <BlockView page={currentPageData} />}
              </>
            ) : (
              <p className="text-sm text-gray-500 text-center mt-8">페이지 데이터를 불러오는 중…</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentViewer;
