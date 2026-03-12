/** File uploader component with drag-and-drop support. */

import React, { useCallback, useRef, useState } from 'react';
import { uploadDocument } from '../services/api';

interface FileUploaderProps {
  onUploaded: (documentId: string, filename: string) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onUploaded }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('PDF 파일만 업로드 가능합니다');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setError('파일 크기가 100MB를 초과합니다');
      return;
    }
    setError(null);
    setUploading(true);
    setUploadProgress(0);
    try {
      const response = await uploadDocument(file, (pct) => setUploadProgress(pct));
      onUploaded(response.document_id, file.name);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '업로드 실패';
      setError(msg);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  }, [onUploaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = '';
  };

  return (
    <div>
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200
          ${isDragging ? 'border-brand-500 bg-brand-950/30' : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/30'}
          ${uploading ? 'cursor-not-allowed' : ''}`}
        role="button"
        aria-label="PDF 파일 업로드 영역"
        id="file-upload-zone"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,application/pdf"
          onChange={handleInputChange}
          className="hidden"
          disabled={uploading}
          id="pdf-file-input"
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-4 border-brand-600 border-t-transparent rounded-full animate-spin" />
            <div className="w-full max-w-xs">
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-brand-500 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
              <p className="text-sm text-gray-400 mt-2">업로드 중… {uploadProgress}%</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors duration-200
              ${isDragging ? 'bg-brand-600/30' : 'bg-gray-800'}`}>
              <svg className="w-8 h-8 text-brand-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div>
              <p className="text-gray-200 font-medium text-base">
                {isDragging ? '여기에 놓으세요' : 'PDF를 드래그하거나 클릭하여 선택'}
              </p>
              <p className="text-gray-500 text-sm mt-1">최대 100MB · PDF 파일만 지원</p>
            </div>
            <button type="button" className="btn-primary pointer-events-none" id="upload-select-btn">
              파일 선택
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-3 flex items-center gap-2 text-red-400 text-sm bg-red-950/30 border border-red-800/50 rounded-xl px-4 py-3">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          {error}
        </div>
      )}
    </div>
  );
};

export default FileUploader;
