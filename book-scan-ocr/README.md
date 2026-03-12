# Book Scan OCR

스캔된 책 PDF를 업로드하면 PaddleOCR로 텍스트를 자동 추출하고, 구조화된 JSON으로 저장 및 웹에서 열람하는 풀스택 웹 애플리케이션입니다.

## 시스템 구성

| 구성 요소 | 기술 |
|-----------|------|
| 백엔드 | Python 3.10+, FastAPI |
| OCR 엔진 | PaddleOCR (한국어) |
| PDF 변환 | pdf2image + poppler |
| 이미지 처리 | OpenCV, Pillow |
| 프론트엔드 | React 18 + TypeScript + Tailwind CSS |
| 번들러 | Vite |

## 사전 설치 (시스템 의존성)

### macOS

```bash
brew install poppler
```

### Ubuntu / Debian

```bash
sudo apt-get install -y poppler-utils
```

### Python 3.10 이상 필요

```bash
python3 --version  # 3.10 이상이어야 합니다
```

## 설치 및 실행

### 1. 백엔드

```bash
cd book-scan-ocr/backend

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# 패키지 설치 (최초 1회, 수 분 소요)
pip install -r requirements.txt

# 서버 실행 (포트 8000)
uvicorn main:app --reload --port 8000
```

> **첫 실행 시 PaddleOCR 한국어 모델을 자동 다운로드합니다 (약 500MB). 인터넷 연결 필요.**

### 2. 프론트엔드

```bash
cd book-scan-ocr/frontend

# 패키지 설치 (최초 1회)
npm install

# 개발 서버 실행 (포트 5173)
npm run dev
```

브라우저에서 http://localhost:5173 접속

---

## API 문서

백엔드 실행 후 http://localhost:8000/docs 에서 Swagger UI를 확인할 수 있습니다.

### 주요 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/upload` | PDF 업로드 및 OCR 시작 |
| GET | `/api/documents` | 문서 목록 |
| GET | `/api/documents/{id}/status` | 처리 상태 조회 |
| GET | `/api/documents/{id}` | 전체 결과 JSON |
| GET | `/api/documents/{id}/pages/{n}` | 특정 페이지 결과 |
| GET | `/api/documents/{id}/pages/{n}/image` | 페이지 이미지 |
| GET | `/api/documents/{id}/download` | 결과 JSON 다운로드 |
| GET/POST | `/api/settings` | 설정 조회/저장 |

---

## 환경변수 (선택)

`backend/.env` 파일을 생성하여 설정을 오버라이드할 수 있습니다:

```
UPLOAD_DIR=/path/to/uploads
PROCESSED_DIR=/path/to/processed
DEFAULT_DPI=300
MAX_FILE_SIZE_MB=100
USE_GPU=false
```

---

## 테스트 실행

```bash
cd backend
source venv/bin/activate
pytest tests/ -v
```

---

## 파일 구조

```
book-scan-ocr/
├── backend/
│   ├── main.py                  # FastAPI 앱 진입점
│   ├── config.py                # 환경설정
│   ├── requirements.txt
│   ├── routers/
│   │   ├── documents.py         # 문서 API
│   │   └── settings.py          # 설정 API
│   ├── services/
│   │   ├── ocr_service.py       # PaddleOCR 처리
│   │   ├── pdf_service.py       # PDF → 이미지 변환
│   │   └── storage_service.py   # 파일 저장/조회
│   ├── models/
│   │   ├── document.py          # Pydantic 모델
│   │   └── settings.py          # 설정 모델
│   ├── uploads/                 # 업로드된 PDF
│   ├── processed/               # OCR 결과 저장소
│   │   └── {document_id}/
│   │       ├── images/          # 페이지 PNG
│   │       ├── pages/           # 페이지 JSON
│   │       ├── result.json      # 전체 결과
│   │       └── meta.json        # 진행 상태
│   └── tests/
├── frontend/
│   └── src/
│       ├── App.tsx              # 라우팅
│       ├── components/          # React 컴포넌트
│       ├── hooks/               # 커스텀 훅
│       ├── services/            # API 클라이언트
│       └── types/               # TypeScript 타입
└── README.md
```

---

## JSON 출력 스키마 예시

```json
{
  "document_id": "...",
  "filename": "책이름.pdf",
  "total_pages": 150,
  "ocr_engine": "paddleocr",
  "language": "korean",
  "pages": [
    {
      "page_number": 1,
      "text_blocks": [
        {
          "text": "제1장 선사시대",
          "confidence": 0.95,
          "bbox": { "top_left": [120, 80], ... }
        }
      ],
      "full_text": "제1장 선사시대\n..."
    }
  ]
}
```
