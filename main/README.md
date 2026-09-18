# 중고 의류 재판매 가치 판별 서비스

> Azure 기반 AI 의류 검색 & 재판매 가격 산정 서비스
> Docker 컨테이너 기반 배포, Azure Portal 콘솔 배포 지원

---

## 📋 프로젝트 개요

트렌드 기반 중고 의류 유사도 검색 및 재판매 가능 여부를 AI로 자동 판별하는 서비스입니다.

### 주요 기능
1. **무신사 상품 크롤링** - 카테고리별 상품 데이터 수집 (상의, 아우터, 바지, 원피스/스커트)
2. **객체 탐지 + 임베딩** - Fashion Object Detection → CLIP 임베딩 → ChromaDB 저장
3. **유사 상품 검색** - 업로드 이미지와 유사한 무신사 상품 Top-K 검색
4. **재판매 정보 생성** - Azure Custom Vision (오염 탐지) + Azure OpenAI (판매 정보 자동 생성)

---

## 🗂️ 프로젝트 구조 

```
clothing-resale-project/
├── README.md                          # 이 파일
├── ARCHITECTURE.md                    # Azure 아키텍처 문서
│
├── crawler/                           # 무신사 크롤러
│   ├── musinsa_crawler.py            
│   ├── Dockerfile                     # 크롤러 Docker 이미지
│   └── requirements.txt
│
├── embedding/                         # ChromaDB 임베딩 파이프라인
│   ├── musinsa_to_chromadb.py        
│   ├── Dockerfile                     # 임베딩 Docker 이미지
│   └── requirements.txt
│
├── search-app/                        # 유사 상품 검색 앱
│   ├── musinsa_detect.py             
│   ├── Dockerfile                     # 검색 앱 Docker 이미지
│   └── requirements.txt
│
├── main-app/                          # 통합 메인 앱 (최종 UI)
│   ├── app.py                        # Streamlit 메인 앱
│   ├── Dockerfile
│   └── requirements.txt
│
├── chromadb-server/                   # ChromaDB 서버
│   ├── Dockerfile                     # ChromaDB 서버 Docker 이미지
│   └── docker-compose.yml            # 로컬 테스트용
│
├── docker-compose.yml                 # 전체 서비스 로컬 테스트
│
└── azure-deploy/                      # Azure 배포 스크립트
    ├── deploy-quide.md
```

---

## 🐳 Docker 기반 배포 전략

### 1. **ChromaDB 서버** (Azure VM + Docker)
```bash
# Azure VM에 ChromaDB 서버 배포
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v /mnt/chromadb-data:/chroma/data \
  chromadb/chroma:latest
```

### 2. **크롤러** (Azure Container Instances - 스케줄 실행)
```bash
# 매주 1회 실행 (Azure Logic Apps 트리거)
docker build -t crawler:latest ./crawler
docker run crawler:latest
```

### 3. **임베딩 파이프라인** (Azure Container Instances - 수동/스케줄 실행)
```bash
# 크롤링 후 실행
docker build -t embedding:latest ./embedding
docker run \
  -e CHROMADB_HOST=<VM_IP> \
  -v /data:/data \
  embedding:latest
```

### 4. **검색 앱** (Azure Container Apps - 항상 실행)
```bash
# 원본 musinsa_detect.py 기반
docker build -t search-app:latest ./search-app
```

### 5. **메인 앱** (Azure Container Apps - 항상 실행)
```bash
# 통합 UI (검색 + 오염 탐지 + 정보 생성)
docker build -t main-app:latest ./main-app
```

---

## 🚀 빠른 시작 (로컬 테스트)

### 1. 전체 서비스 Docker Compose로 실행
```bash
# 모든 서비스 한 번에 실행
docker-compose up -d

# 서비스 확인
docker-compose ps

# ChromaDB: http://localhost:8000
# 검색 앱: http://localhost:8501
# 메인 앱: http://localhost:8502
```

### 2. 개별 서비스 실행

#### 크롤러 실행
```bash
cd crawler
docker build -t musinsa-crawler .
docker run -v $(pwd)/data:/data musinsa-crawler
```

#### 임베딩 실행
```bash
cd embedding
docker build -t musinsa-embedding .
docker run \
  -e CHROMADB_HOST=host.docker.internal \
  -v $(pwd)/data:/data \
  musinsa-embedding
```

#### 검색 앱 실행
```bash
cd search-app
docker build -t musinsa-search .
docker run -p 8501:8501 \
  -e CHROMADB_HOST=host.docker.internal \
  musinsa-search
```

---

## ☁️ Azure 배포 가이드 (콘솔 기반)

### 전제 조건
- Azure 구독 (Free Tier 가능)
- Azure Container Registry (ACR) 생성
- Azure Portal 접근 권한

### Step 1: Azure Container Registry 생성
```bash
# Azure Portal에서 수동 생성
1. Portal → "Container Registries" 검색
2. "+ Create" 클릭
3. Resource Group: clothing-resale-rg
4. Registry Name: clothingresaleacr
5. Location: Korea Central
6. SKU: Basic
```

### Step 2: 로컬에서 이미지 빌드 & ACR 푸시
```bash
# ACR 로그인
az login
az acr login --name clothingresaleacr

# 이미지 빌드 & 푸시
cd crawler
docker build -t clothingresaleacr.azurecr.io/crawler:latest .
docker push clothingresaleacr.azurecr.io/crawler:latest

cd ../embedding
docker build -t clothingresaleacr.azurecr.io/embedding:latest .
docker push clothingresaleacr.azurecr.io/embedding:latest

cd ../search-app
docker build -t clothingresaleacr.azurecr.io/search-app:latest .
docker push clothingresaleacr.azurecr.io/search-app:latest

cd ../main-app
docker build -t clothingresaleacr.azurecr.io/main-app:latest .
docker push clothingresaleacr.azurecr.io/main-app:latest
```

### Step 3: ChromaDB VM 배포 (Azure Portal)
```
1. Portal → "Virtual Machines" → "+ Create"
2. 설정:
   - Name: chromadb-vm
   - Size: Standard_E4s_v3 (4 vCPU, 32GB RAM)
   - OS: Ubuntu 22.04 LTS
   - Disk: Premium SSD 256GB
3. SSH 접속 후:
   sudo apt update && sudo apt install -y docker.io
   sudo docker run -d --name chromadb -p 8000:8000 -v /mnt/chromadb-data:/chroma/data chromadb/chroma:latest
```

### Step 4: Container Apps 배포 (Azure Portal)

#### 검색 앱 배포
```
1. Portal → "Container Apps" → "+ Create"
2. 설정:
   - App name: musinsa-search-app
   - Container image: clothingresaleacr.azurecr.io/search-app:latest
   - CPU: 1 vCPU, Memory: 2GB
   - Ingress: Enabled (External, Port 8501)
   - Environment Variables:
     * CHROMADB_HOST: <VM_Private_IP>
     * CHROMADB_PORT: 8000
3. Review + Create
```

#### 메인 앱 배포
```
1. Portal → "Container Apps" → "+ Create"
2. 설정:
   - App name: clothing-resale-main
   - Container image: clothingresaleacr.azurecr.io/main-app:latest
   - CPU: 1 vCPU, Memory: 2GB
   - Ingress: Enabled (External, Port 8502)
   - Environment Variables:
     * CHROMADB_HOST: <VM_Private_IP>
     * CHROMADB_PORT: 8000
     * AZURE_CUSTOM_VISION_ENDPOINT: <YOUR_ENDPOINT>
     * AZURE_CUSTOM_VISION_KEY: <YOUR_KEY>
     * AZURE_OPENAI_ENDPOINT: <YOUR_ENDPOINT>
     * AZURE_OPENAI_KEY: <YOUR_KEY>
3. Review + Create
```

### Step 5: Container Instances로 크롤러/임베딩 실행
```
1. Portal → "Container Instances" → "+ Create"
2. 크롤러 설정:
   - Name: musinsa-crawler-job
   - Container image: clothingresaleacr.azurecr.io/crawler:latest
   - Restart Policy: Never (일회성 실행)
   - Volumes: Azure Files (크롤링 결과 저장)

3. 임베딩 설정:
   - Name: musinsa-embedding-job
   - Container image: clothingresaleacr.azurecr.io/embedding:latest
   - Environment Variables:
     * CHROMADB_HOST: <VM_Private_IP>
   - Volumes: Azure Files (크롤링 결과 읽기)
```

---

## 📊 Azure 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Cloud                                  │
│                                                                   │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │  Container       │     │  Container       │                  │
│  │  Registry (ACR)  │────▶│  Apps            │                  │
│  │                  │     │  (검색 앱)       │◀──┐              │
│  └──────────────────┘     └──────────────────┘   │              │
│                                                    │              │
│  ┌──────────────────┐     ┌──────────────────┐   │              │
│  │  Container       │     │  VM (Ubuntu)     │   │              │
│  │  Instances       │     │                  │   │              │
│  │  (크롤러/임베딩) │────▶│  ChromaDB Server │◀──┤              │
│  └──────────────────┘     │  (Docker)        │   │              │
│                            └──────────────────┘   │              │
│                                                    │              │
│  ┌──────────────────┐     ┌──────────────────┐   │              │
│  │  Azure           │     │  Container       │   │              │
│  │  Custom Vision   │────▶│  Apps            │───┘              │
│  │  (오염 탐지)     │     │  (메인 앱)       │                  │
│  └──────────────────┘     └──────────────────┘                  │
│                                                                   │
│  ┌──────────────────┐                                            │
│  │  Azure OpenAI    │                                            │
│  │  (GPT-4o)        │                                            │
│  └──────────────────┘                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💰 예상 비용 (월별)

| 리소스 | 구성 | 월 비용 |
|--------|------|---------|
| **ChromaDB VM** | Standard_E4s_v3 (4 vCPU, 32GB RAM) | $180 |
| **Container Apps (검색)** | 1 vCPU, 2GB (항상 실행) | $30 |
| **Container Apps (메인)** | 1 vCPU, 2GB (항상 실행) | $30 |
| **Container Instances (크롤러)** | 주 1회 실행 (1시간) | $5 |
| **Container Instances (임베딩)** | 주 1회 실행 (2시간) | $10 |
| **Azure Custom Vision** | Standard S0 (1,000건/월) | $50 |
| **Azure OpenAI** | GPT-4o (1,000건/월) | $50 |
| **Azure Container Registry** | Basic | $5 |
| **Azure Files** | 100GB (데이터 저장) | $20 |
| **총 예상 비용** | | **$380/월** |

---

## 🔧 원본 코드 설명

### 1. `musinsa_crawler.py`
- **기능**: 무신사 상품 크롤링 (Selenium)
- **카테고리**: 상의, 아우터, 바지, 원피스/스커트
- **출력**: JSON 파일 (상품ID, 브랜드, 제품명, 가격, 이미지URL 등)

### 2. `musinsa_to_chromadb.py`
- **기능**: 
  1. JSON 데이터 로드
  2. Fashion Object Detection (yainage90 모델)
  3. 크롭된 이미지를 FashionCLIP으로 임베딩
  4. ChromaDB에 저장 (카테고리별 컬렉션)
- **특징**: Streamlit UI 포함 (진행 상황 실시간 표시)

### 3. `musinsa_detect.py`
- **기능**: 
  1. 이미지 업로드
  2. Fashion Object Detection (의류 영역 감지)
  3. ChromaDB에서 유사 상품 검색
  4. 결과 표시 (유사도 점수 포함)
- **특징**: Streamlit 기반 웹 앱

---

## 🔬 Retrieval 평가 시스템 (Milestone 1–10)

`musinsa_to_chromadb.py`/`musinsa_detect.py`/`app.py`에 흩어져 있던 detection→crop→embedding→
search 로직을 `retrieval/`(UI 독립적인 core 로직)와 `evaluation/`(재현 가능한 실험/평가 인프라)로
분리하고, RAW 임베딩과 여러 BBox selection policy·category filter policy·padding 값을
동일한 평가셋·지표로 비교할 수 있는 ablation 인프라를 구축했습니다. 상세 설계 근거와 스펙은
`IMPLEMENTATION_SPEC.md`를 따릅니다.

### 무엇이 구축되어 있는가

| 영역 | 위치 | 내용 |
|---|---|---|
| Retrieval core | `retrieval/` | detection, bbox selection/crop, category mapping, embedding, search — UI와 완전히 분리 |
| 평가 지표 | `evaluation/metrics.py` | Precision@K, Recall@K, MRR, NDCG@K, incompatible category rate, relevant exclusion rate (dependency-free) |
| Ablation 실험 | `evaluation/run_*.py` | E0(RAW) ~ E8(RAW+soft filter)까지 9개 실험, config는 `evaluation/configs/*.json` |
| Failure 분석 | `evaluation/failure_analysis.py` | 10개 failure taxonomy, poor-query 자동 추출, 실험별 failure distribution |
| Sensitivity 지표 | `evaluation/sensitivity.py` | detection threshold / padding / Top-K sweep을 비교하기 위한 지표 |
| Golden set / Regression | `evaluation/golden_set.py`, `evaluation/regression.py`, `tests/regression/` | golden set 커버리지 검증, quality regression 임계값 체크, critical query rank 체크 |
| 결정 기록 | `docs/decisions/ADR-*.md`, `docs/decisions/final_config.yaml` | RAW vs BBox, bbox policy, category filter, padding, 최종 pipeline에 대한 ADR |
| 마일스톤별 근거 | `docs/evidence/milestone-*.md` | 각 마일스톤의 설계 결정·대안·검증 커맨드/결과·알려진 한계 기록 |

### 현재 상태 (정직하게)

- **실제 데이터로 실험을 실행한 적이 없습니다.** `evaluation/dataset/labels.json`은 아직 빈
  스캐폴드(`{"queries": {}}`)이고, 이 개발 환경에는 `torch`/`chromadb`/`transformers`/
  `open_clip`이 설치되어 있지 않습니다. 모든 실험 러너와 지표는 fake/synthetic 입력으로만
  검증되었습니다 (`cd main && python -m pytest tests/ -q`).
- 따라서 `docs/decisions/final_config.yaml`의 모든 값은 **TBD**입니다 — 실험 근거 없이 최종
  pipeline을 확정하지 않는다는 원칙(`IMPLEMENTATION_SPEC.md` section 50)을 지켰습니다.
- 실제 데이터셋과 ML 의존성이 준비되면, 이미 만들어진 인프라(`run_bbox_experiments.py` 등)를
  그대로 실행해 `docs/decisions/ADR-*.md`의 Decision을 채우는 것이 다음 단계입니다.

---

## 🎯 다음 단계

### Phase 1: 로컬 테스트
- [ ] Docker Compose로 전체 서비스 실행
- [ ] 크롤러 → 임베딩 → 검색 파이프라인 검증

### Phase 2: Azure 배포
- [ ] ChromaDB VM 설정
- [ ] ACR에 이미지 푸시
- [ ] Container Apps 배포 (검색 앱)
- [ ] Container Apps 배포 (메인 앱)

### Phase 3: 통합 & 자동화
- [ ] Azure Logic Apps로 크롤러 스케줄링
- [ ] Azure Monitor로 모니터링 설정
- [ ] Azure Key Vault로 시크릿 관리

---

## 📞 문의

- **프로젝트**: Sookmyung Women's University 인공지능산업체특강
- **기술 스택**: Azure, Docker, Streamlit, ChromaDB, CLIP, Object Detection, Custom Vision

---

## 📝 참고 사항

- Docker 이미지 빌드 시 모델 다운로드로 인해 시간이 걸릴 수 있음 (최초 1회)
- ChromaDB 데이터는 Azure Files 또는 VM 디스크에 영구 저장 필요
