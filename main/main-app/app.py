"""
중고 의류 재판매 가치 판별 서비스 - 메인 앱
Azure OpenAI + ChromaDB 연동 버전
"""

import streamlit as st
import pandas as pd
from PIL import Image
import io
import os
import sys
import logging
from openai import AzureOpenAI
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import search as retrieval_search
from retrieval.config import (
    COLLECTION_NAMES,
    DETECTION_MODEL,
    DETECTION_THRESHOLD,
    EMBEDDING_MODEL,
    MIN_BBOX_AREA,
)
from retrieval.detection import detect_fashion_items as core_detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model as core_load_detection_model
from retrieval.models import load_embedding_model as core_load_embedding_model
from retrieval.preprocessing import preprocess_image

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수
CHROMADB_HOST = os.getenv("CHROMADB_HOST", None)
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", 8000))
AZURE_CUSTOM_VISION_ENDPOINT = os.getenv("AZURE_CUSTOM_VISION_ENDPOINT", "")
AZURE_CUSTOM_VISION_KEY = os.getenv("AZURE_CUSTOM_VISION_KEY", "")
AZURE_CUSTOM_VISION_PROJECT_ID = os.getenv("AZURE_CUSTOM_VISION_PROJECT_ID", "")

# Azure OpenAI 환경 변수
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# OpenAI 클라이언트 초기화
openai_client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY:
    try:
        openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI 클라이언트 초기화 완료")
    except Exception as e:
        logger.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")

# 세션 상태 초기화
if 'detection_result' not in st.session_state:
    st.session_state.detection_result = None
if 'similar_items' not in st.session_state:
    st.session_state.similar_items = None

def get_chromadb_client():
    """ChromaDB 클라이언트 가져오기 (환경에 따라 자동 전환)"""
    if CHROMADB_HOST:
        logger.info(f"ChromaDB 원격 서버 연결: {CHROMADB_HOST}:{CHROMADB_PORT}")
    else:
        logger.info("ChromaDB 로컬 파일 사용: ./musinsa_fashion_db_crop")
    return retrieval_search.get_chromadb_client(host=CHROMADB_HOST, port=CHROMADB_PORT)

@st.cache_resource
def load_detection_model():
    """객체 탐지 모델 로드"""
    try:
        image_processor, model, device = core_load_detection_model(DETECTION_MODEL)
        logger.info(f"객체 탐지 모델이 {device}에 로드되었습니다.")
        return image_processor, model, device
    except Exception as e:
        logger.error(f"객체 탐지 모델 로드 중 오류: {e}")
        return None, None, None

def detect_defects_azure(image_bytes):
    """Azure Custom Vision으로 오염/손상 탐지"""
    if not AZURE_CUSTOM_VISION_ENDPOINT or not AZURE_CUSTOM_VISION_KEY:
        # Azure 설정이 없으면 더미 데이터 반환
        return {
            "is_resellable": True,
            "defects": [],
            "status": "판매 가능 (Azure Custom Vision 미설정)"
        }
    
    try:
        from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
        from msrest.authentication import ApiKeyCredentials
        
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": AZURE_CUSTOM_VISION_KEY})
        predictor = CustomVisionPredictionClient(AZURE_CUSTOM_VISION_ENDPOINT, credentials)
        
        results = predictor.detect_image(
            AZURE_CUSTOM_VISION_PROJECT_ID,
            "Iteration1",
            image_bytes
        )
        
        defects = []
        for pred in results.predictions:
            if pred.probability > 0.90 and pred.tag_name in ["Pollution", "Damage"]:
                defects.append({
                    "type": pred.tag_name,
                    "confidence": pred.probability
                })
        
        return {
            "is_resellable": len(defects) == 0,
            "defects": defects,
            "status": "판매 가능" if len(defects) == 0 else "판매 불가"
        }
    except Exception as e:
        logger.error(f"Azure Custom Vision 오류: {e}")
        return {
            "is_resellable": True,
            "defects": [],
            "status": "판매 가능 (오류 발생)"
        }

def detect_fashion_items(image, threshold=DETECTION_THRESHOLD):
    """패션 아이템 탐지"""
    image_processor, detection_model, device = load_detection_model()

    if image_processor is None:
        return []

    try:
        detected_items = core_detect_fashion_items(
            image,
            image_processor=image_processor,
            model=detection_model,
            device=device,
            threshold=threshold,
            min_area=MIN_BBOX_AREA,
        )
        detected_items.sort(key=lambda x: x['area'], reverse=True)
        return detected_items

    except Exception as e:
        logger.error(f"패션 아이템 탐지 중 오류: {e}")
        return []

def search_similar_items(image, top_k=5):
    """ChromaDB에서 유사 상품 검색"""
    try:
        # 환경에 따라 자동 전환
        client = get_chromadb_client()

        # CLIP 모델로 직접 임베딩 생성 (기존 동작 유지: 매 호출마다 새로 로드)
        clip_model, preprocess_val, clip_device = core_load_embedding_model(EMBEDDING_MODEL)
        query_embedding = embed_image(image, clip_model, preprocess_val, clip_device)

        # 모든 컬렉션에서 검색 (기존 동작 유지: id 기준 중복 제거 없음)
        results = retrieval_search.search_collections(
            client,
            COLLECTION_NAMES,
            query_embedding=query_embedding,
            top_k=top_k,
            dedupe=False,
        )

        similar_items = []
        for result in results:
            item_data = dict(result['metadata'])
            item_data['similarity_score'] = 1 / (1 + result['raw_distance'])
            item_data['distance'] = result['raw_distance']
            item_data['collection'] = result['collection']
            similar_items.append(item_data)

        return similar_items

    except Exception as e:
        logger.error(f"유사 상품 검색 중 오류: {e}")
        return []

def image_to_base64(image):
    """이미지를 base64로 인코딩"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_product_info_with_llm(uploaded_image, similar_items, defect_info):
    """LLM을 사용한 상품 정보 자동 생성"""
    
    # OpenAI 클라이언트가 없으면 기본 템플릿 사용
    if not openai_client:
        logger.warning("Azure OpenAI 미설정 - 템플릿 기반 생성")
        return generate_product_info_template(similar_items)
    
    try:
        # 업로드 이미지를 base64로 인코딩
        image_base64 = image_to_base64(uploaded_image)
        
        # 유사 상품 정보 포맷팅
        similar_info = ""
        for idx, item in enumerate(similar_items[:3], 1):
            similar_info += f"""
{idx}. 상품명: {item.get('name', '알 수 없음')}
   브랜드: {item.get('brand', '알 수 없음')}
   가격: {item.get('price', '알 수 없음')}원
   유사도: {item['similarity_score']:.2%}
   카테고리: {item.get('collection', '알 수 없음')}
"""
        
        # 프롬프트 구성
        system_prompt = """당신은 중고 의류 판매글을 자동으로 생성하는 AI입니다.
입력은 JSON 배열(list)이며, 규칙은 다음과 같습니다.

- 배열의 1번째 요소(인덱스 0): 판매자가 올릴 '대상 상품'
- 배열의 1번째 요소(인덱스 0)를 제외한 나머지 요소들: 유사 상품(관련성 높은 상품) 목록

작업:
1) 대상 상품의 brand, name, category, price, discount_rate 정보를 파악합니다.
2) 유사 상품들의 price/brand/name/category와 similarity_score를 참고하여
   - 시장 가격 범위(유사 상품 가격대)
   - 적절한 중고 판매가(추천가 1개 + 협의 가능 범위)
를 제안합니다.
3) 허위 사실은 쓰지 말고, 입력에 없는 상태/사이즈/착용횟수 등은 '확실하지 않음'으로 처리하거나 언급하지 마세요.
4) 최종 출력은 반드시 아래 JSON만 출력하세요.

{
  "title": "...",
  "content": "..."
}

제약:
- JSON 외 다른 텍스트 금지.
- title, content 필드만 포함.

반드시 지켜야 할 책임 원칙:
- 입력에 없는 정보(사이즈, 사용감, 착용감, 손상 정도, 구성품)는 임의로 생성하지 말고,
  확실하지 않은 부분은 글에 포함하지 않습니다.

- 성별, 연령, 체형 등을 단정하거나 암시하는 문구는 작성하지 않습니다
  (예: 여성용일 것 같다는 추측 금지).

- 특정 브랜드 혹은 특정 스타일에 대한 편향적·우월적 표현은 사용하지 않습니다.
  (예: '이 브랜드는 무조건 고급' 등 단정 금지)

- 사실과 다르거나 과장된 표현은 금지합니다.
  (예: "새상품급!", "사용감 전혀 없음!" 등 입력 없이 사용 금지)

- 설명은 구매자가 오해하지 않도록 중립적이고 객관적으로 작성합니다."""

        user_prompt = f"""
**업로드된 상품 상태:**
- 오염/손상: {"없음 (재판매 가능)" if defect_info['is_resellable'] else "있음 (재판매 불가)"}

**무신사 트렌드 유사 상품 Top 3:**
{similar_info}
"""

        # Azure OpenAI API 호출
        response = openai_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
            max_tokens=800
        )
        
        # 응답 파싱
        content = response.choices[0].message.content
        
        # 제목과 설명 분리
        if "---제목---" in content and "---설명---" in content:
            parts = content.split("---제목---")[1].split("---설명---")
            title = parts[0].strip()
            description = parts[1].strip()
        else:
            # 파싱 실패 시 전체를 설명으로
            title = "✨ 트렌디한 중고 의류 - 무신사 스타일"
            description = content
        
        logger.info("LLM으로 상품 정보 생성 완료")
        
        return {
            "title": title,
            "description": description,
            "generated_by": f"Azure OpenAI ({AZURE_OPENAI_DEPLOYMENT})"
        }
        
    except Exception as e:
        logger.error(f"LLM 생성 중 오류: {e}")
        # 에러 시 템플릿 사용
        return generate_product_info_template(similar_items)

def generate_product_info_template(similar_items):
    """템플릿 기반 상품 정보 생성 (fallback)"""
    if not similar_items:
        return {
            "title": "트렌디한 중고 의류",
            "description": "인기 브랜드 중고 의류입니다.",
            "generated_by": "Template"
        }
    
    top_item = similar_items[0]
    brand = top_item.get('brand', '알 수 없음')
    category = top_item.get('collection', '의류')
    
    title = f"✨ [{brand}] 트렌디한 {category} - 무신사 인기 스타일"
    
    description = f"""
**트렌드 분석 기반 상품 설명**

- **카테고리**: {category}
- **유사 브랜드**: {brand}
- **유사도**: {top_item.get('similarity_score', 0):.2%}

**추천 이유**:
현재 무신사에서 인기 있는 스타일과 {top_item.get('similarity_score', 0):.2%}의 높은 유사도를 보입니다.
트렌드를 반영한 디자인으로 재판매 가치가 높습니다.

**상태**: 오염 및 손상 없는 깨끗한 상태
"""
    
    return {
        "title": title,
        "description": description,
        "generated_by": "Template"
    }

# === 메인 앱 ===
st.set_page_config(
    page_title="중고 의류 재판매 가치 판별",
    page_icon="👚",
    layout="wide"
)

# 사이드바
with st.sidebar:
    st.title("💡 프로젝트 개요")
    st.markdown("""### 👚중고 의류 재판매 가치 판별 서비스
이 서비스는 **중고 의류 판매자를 위한 AI 기반 보조 도구**로, 상품의 *재판매 가치 판별 → 트렌드 분석 → 상품 판매글 자동 생성*까지 판매 준비 과정을 빠르고 간편하게 도와줍니다.

---

### 🌟 제공 기능
**1) 오염·이염 탐지 (Azure Custom Vision 기반)**  
- 업로드한 이미지에서 오염/이염 여부를 감지하여  
  **재판매 가능 여부를 자동 판단**합니다.

**2) 트렌드 유사 상품 매칭 (FashionCLIP + ChromaDB)**  
- 현재 인기 있는 무신사 베스트 상품과의 유사도를 분석하여  
  **트렌드 적합성을 확인**할 수 있습니다.

**3) AI 기반 상품 제목·설명 자동 생성**  
- Azure OpenAI가 유사 상품 정보를 기반으로  
  **판매에 바로 사용할 수 있는 상품명과 설명을 생성**합니다.

---

### 💡 기대 효과
- 상품 등록 시간을 단축할 수 있습니다.  
- 판매 문구 작성에 어려움을 느끼는 판매자에게 도움을 줍니다.  
- 트렌드 기반 유사 상품 비교로 **경쟁력 있는 가격·문구 설정**이 가능합니다.  """)
    
    st.markdown("---")
    st.subheader("모델 정보")
    st.text("* 오염 탐지: Azure Custom Vision")
    st.text("* 임베딩: FashionCLIP")
    st.text("* 벡터 DB: ChromaDB")
    st.text("* 정보 생성: Azure OpenAI")
    
    # 연결 상태 표시
    st.markdown("---")
    st.subheader("연결 상태")
    if openai_client:
        st.success("✅ Azure OpenAI 연결됨")
    else:
        st.warning("⚠️ Azure OpenAI 미연결 (템플릿 모드)")
    
    if CHROMADB_HOST:
        st.success(f"✅ ChromaDB: {CHROMADB_HOST}")
    else:
        st.info("📁 ChromaDB: 로컬 파일")
    
    top_k = st.slider("유사 상품 Top-K 개수", 1, 10, 3)

# 메인 타이틀
st.title("👚 중고 의류 재판매 가치 판별 서비스")

# Step 1: 이미지 업로드
st.header("1. 의류 이미지 업로드")

# 포용성 - 촬영 가이드 제공
with st.expander("📷 중고 의류 촬영 가이드 (분석 정확도 향상을 위해 권장)", expanded=False):
    st.markdown(
        """
- **배경**: 가능하면 **단색 배경**에서 촬영해주세요. (침대시트/바닥 패턴 최소화)
- **구도**: 옷의 **전체 실루엣이 화면에 모두 나오도록** 촬영해주세요.
- **밝기**: 충분한 조명 아래에서 촬영하고, 너무 어둡거나 과도하게 밝지 않게 조정해주세요.
- **구김/접힘**: 옷은 최대한 **펼쳐진 상태**에서 촬영하면 오염·이염 탐지 정확도가 올라갑니다.
        """
    )

uploaded_file = st.file_uploader(
    "판매할 중고 의류 이미지를 선택해주세요.",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 의류 이미지', use_column_width=True)
    
    # 개인정보보호 문구
    st.caption("※ 업로드한 이미지는 분석 후 즉시 삭제되며, 서버에 저장되지 않습니다.")
    st.markdown("---")
    
    # Step 2: 오염 탐지
    st.header("2. 오염·이염 탐지 및 재판매 판별")
    
    with st.spinner("🔍 Azure Custom Vision으로 결함 탐지 중..."):
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        detection_result = detect_defects_azure(img_bytes.read())
        st.session_state.detection_result = detection_result
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if detection_result['is_resellable']:
            st.success("🟢 재판매 가능")
            st.metric("판단 근거", "오염/이염 없음")
        else:
            st.error("🔴 재판매 불가")
            defect_info = detection_result['defects'][0]
            st.metric(
                "판단 근거",
                f"{defect_info['type']} 감지 ({defect_info['confidence']:.2%})"
            )
        
        # 책임성 문구
        st.caption("※ 본 결과는 AI 분석 기준이며, 실제 상품 상태는 사용자가 반드시 직접 확인해야 합니다.")
    
    with col2:
        st.info("💡 Azure Custom Vision 모델이 의류 이미지를 분석하여 오염 및 손상을 탐지했습니다.")
    
    st.markdown("---")
    
    if not detection_result['is_resellable']:
        st.warning("⚠️ 오염이 감지되어 이후 **트렌드 분석 및 상품 정보 자동 생성 기능**은 비활성화됩니다.")
    else:
        # Step 3: 객체 탐지
        with st.spinner("의류 영역 감지 중..."):
            detected_items = detect_fashion_items(image)

        # 가장 큰 영역 자동 선택 (bbox 미탐지 시 원본 이미지로 fallback)
        cropped_image, _preprocessing_metadata = preprocess_image(
            image, detected_items, policy="largest", fallback_policy="raw"
        )
        
        # Step 4: 유사 상품 검색
        st.header(f"3. 트렌드 유사 상품 (Top-{top_k})")
        
        with st.spinner("🔍 ChromaDB에서 유사 상품 검색 중..."):
            similar_items = search_similar_items(cropped_image, top_k=top_k)
            st.session_state.similar_items = similar_items
        
        if similar_items:
            df_similar = pd.DataFrame([
                {
                    '상품명': item.get('name', '알 수 없음')[:50],
                    '브랜드': item.get('brand', '알 수 없음'),
                    '가격': f"{item.get('price', '알 수 없음')}원",
                    '유사도': f"{item['similarity_score']:.2%}",
                    'URL': item.get('product_url', '')
                }
                for item in similar_items
            ])
            st.dataframe(df_similar, use_container_width=True)
            
            # 크롤링 출처 (투명성·책임성)
            st.caption("※ 유사 상품 정보는 무신사 베스트 상품 데이터를 기반으로 수집되었습니다.")
        else:
            st.warning("유사한 상품을 찾지 못했습니다.")
        
        st.markdown("---")
        
        # Step 5: AI 정보 생성
        st.header("4. AI 기반 상품 판매 정보 생성")
        
        with st.spinner("🤖 Azure OpenAI로 상품 정보 생성 중..."):
            product_info = generate_product_info_with_llm(
                cropped_image, 
                similar_items,
                st.session_state.detection_result
            )
        
        # 생성 방식 표시
        if product_info.get('generated_by') == 'Template':
            st.warning("⚠️ Azure OpenAI 미설정 - 템플릿 기반 생성")
        else:
            st.info(f"✨ {product_info.get('generated_by', 'AI')}로 생성됨")
        
        st.subheader("📝 자동 생성된 상품 제목")
        st.text_area("제목:", product_info['title'], height=80, key="title_output")
        
        st.subheader("📄 자동 생성된 상세 설명")
        st.text_area("상세 설명:", product_info['description'], height=300, key="desc_output")
        
        st.markdown("---")
        st.success("재판매 가능 판정 및 상품 정보 생성이 완료되었습니다. 판매 전 실제 상품 상태를 꼭 확인해주세요!")

else:
    st.info("👆 이미지를 업로드하여 시작하세요")

# 하단 책임성 문구
st.markdown("---")
st.caption(
    "이 서비스는 중고 의류 판매를 돕기 위한 **AI 보조 도구**입니다.  \n"
    "AI의 판단 및 추천 문구는 참고용이며, 실제 거래에 대한 **법적·실질적 책임은 사용자에게 있습니다.**  \n"
)