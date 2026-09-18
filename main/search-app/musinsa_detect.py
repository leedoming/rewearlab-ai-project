"""
무신사 패션 이미지 유사도 검색 앱
카테고리별 컬렉션에서 유사 아이템 검색

Detection / bbox crop / ChromaDB search now delegate to the shared
retrieval core (main/retrieval/) so this file only handles Streamlit UI.
"""

import logging
import os
import sys

import streamlit as st
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

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
from retrieval.models import load_detection_model as core_load_detection_model
from retrieval.preprocessing import crop_image as core_crop_image
from retrieval.preprocessing import preprocess_image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musinsa_search.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 세션 상태 초기화
if 'image' not in st.session_state:
    st.session_state.image = None
if 'detected_boxes' not in st.session_state:
    st.session_state.detected_boxes = None
if 'selected_box_index' not in st.session_state:
    st.session_state.selected_box_index = None

@st.cache_resource
def load_detection_model():
    """객체 탐지 모델 로드"""
    try:
        logger.info("객체 탐지 모델 로딩 중...")
        image_processor, model, device = core_load_detection_model(DETECTION_MODEL)
        logger.info(f"객체 탐지 모델이 {device}에 로드되었습니다.")
        return image_processor, model, device
    except Exception as e:
        logger.error(f"객체 탐지 모델 로드 중 오류 발생: {str(e)}")
        raise

def crop_image(image, box):
    """바운딩 박스에 맞게 이미지 크롭"""
    return core_crop_image(image, box)

def resolve_fallback_image(image, detected_items):
    """탐지된 아이템이 없을 때 공유 fallback 메커니즘으로 원본 이미지를 사용한다.

    가짜 detection을 만들지 않고, fallback_used/fallback_reason이 그대로
    기록된 preprocessing 결과를 반환한다.
    """
    return preprocess_image(image, detected_items, policy="largest", fallback_policy="raw")

def detect_fashion_items(image, min_size=MIN_BBOX_AREA, threshold=DETECTION_THRESHOLD):
    """
    패션 아이템 탐지 및 바운딩 박스 추출

    Args:
        image: PIL Image
        min_size: 최소 크기 (픽셀, area 기준)
        threshold: 탐지 신뢰도 임계값

    Returns:
        탐지된 아이템 리스트 (bbox, label, score, area 포함), 면적 내림차순 정렬.
        정렬/선택은 UI 책임이며 core detect_fashion_items()는 후보만 반환한다.
    """
    try:
        image_processor, model, device = load_detection_model()
        detected_items = core_detect_fashion_items(
            image,
            image_processor=image_processor,
            model=model,
            device=device,
            threshold=threshold,
            min_area=min_size,
        )

        # 기존 동작 유지: 면적 기준 내림차순 정렬 후 사용자가 선택
        detected_items.sort(key=lambda x: x['area'], reverse=True)

        logger.info(f"총 {len(detected_items)}개 아이템 탐지됨")
        return detected_items

    except Exception as e:
        logger.error(f"패션 아이템 탐지 중 오류: {str(e)}")
        return []

def search_similar_items(image, top_k=10, selected_collections=None):
    """
    여러 컬렉션에서 유사 아이템 검색

    Args:
        image: PIL Image (크롭된 이미지)
        top_k: 반환할 결과 수
        selected_collections: 검색할 컬렉션 리스트 (None이면 모든 컬렉션)

    Returns:
        유사 아이템 리스트 (UI 표시용으로 similarity_score가 추가된 딕셔너리)
    """
    try:
        chromadb_host = os.getenv("CHROMADB_HOST", None)
        chromadb_port = int(os.getenv("CHROMADB_PORT", 8000))
        client = retrieval_search.get_chromadb_client(host=chromadb_host, port=chromadb_port)
        if chromadb_host:
            logger.info(f"ChromaDB 원격 서버 연결: {chromadb_host}:{chromadb_port}")
        else:
            logger.info("ChromaDB 로컬 파일 사용: ./musinsa_fashion_db_crop")

        # 임베딩할 때와 동일한 모델 사용
        embedding_function = OpenCLIPEmbeddingFunction(model_name=EMBEDDING_MODEL)

        if selected_collections is None:
            selected_collections = COLLECTION_NAMES

        logger.info(f"검색 대상 컬렉션: {selected_collections}")

        results = retrieval_search.search_collections(
            client,
            selected_collections,
            query_image=image,
            top_k=top_k,
            embedding_function=embedding_function,
        )

        # UI 표시용 필드(similarity_score, distance, collection)를 메타데이터에 병합
        # core search layer는 1/(1+distance) 같은 파생 점수를 반환하지 않으므로
        # 여기(UI 계층)에서 변환한다.
        unique_results = []
        for result in results:
            item_data = dict(result['metadata'])
            similarity_score = 1 / (1 + result['raw_distance'])
            item_data['similarity_score'] = similarity_score * 100
            item_data['distance'] = result['raw_distance']
            item_data['collection'] = result['collection']
            unique_results.append(item_data)

        # 로그에 검색 결과 JSON 형태로 출력
        import json
        logger.info("=" * 80)
        logger.info("검색 결과 (JSON)")
        logger.info("=" * 80)
        logger.info(json.dumps(unique_results, ensure_ascii=False, indent=2))
        logger.info("=" * 80)

        return unique_results

    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        return []

def show_similar_items(similar_items):
    """유사 아이템 표시"""
    if not similar_items:
        st.warning("유사한 아이템을 찾지 못했습니다.")
        return

    st.subheader(f"🔍 유사한 아이템 ({len(similar_items)}개)")

    items_per_row = 3
    for i in range(0, len(similar_items), items_per_row):
        cols = st.columns(items_per_row)
        for j, col in enumerate(cols):
            if i + j < len(similar_items):
                item = similar_items[i + j]
                with col:
                    try:
                        # 이미지 표시
                        if 'image_url' in item:
                            st.image(item['image_url'], width="stretch")
                        elif 'uri' in item:
                            st.image(item['uri'], width="stretch")

                        # 유사도 점수
                        st.markdown(f"**유사도: {item['similarity_score']:.1f}%**")

                        # 상품 정보
                        st.write(f"**브랜드**: {item.get('brand', '알 수 없음')}")

                        name = item.get('name', '알 수 없음')
                        if len(name) > 40:
                            name = name[:37] + "..."
                        st.write(f"**제품명**: {name}")

                        st.write(f"**카테고리**: {item.get('category', '알 수 없음')}")
                        st.write(f"**컬렉션**: {item.get('collection', '알 수 없음')}")
                        st.write(f"**가격**: {item.get('price', '알 수 없음')}원")

                        # 탐지 라벨
                        detected_label = item.get('detected_label', 'original')
                        if detected_label != 'original':
                            st.write(f"**탐지 라벨**: {detected_label}")

                        # 상품 URL
                        if 'product_url' in item and item['product_url']:
                            st.markdown(f"[무신사에서 보기]({item['product_url']})")

                        st.divider()

                    except Exception as e:
                        logger.error(f"아이템 표시 중 오류: {e}")
                        st.error("이 아이템을 표시하는 중 오류가 발생했습니다")

def main():
    st.set_page_config(layout="wide")
    st.title("🛍️ 무신사 패션 이미지 검색")

    st.markdown("""
    ### 사용 방법
    1. 패션 이미지를 업로드하세요
    2. 자동으로 의류 아이템을 감지합니다
    3. 검색할 아이템을 선택하세요
    4. 검색할 카테고리를 선택하세요
    5. 유사한 아이템을 찾습니다
    """)

    # 사이드바 옵션
    with st.sidebar:
        st.header("검색 옵션")

        # 검색할 컬렉션 선택
        st.subheader("검색 카테고리")
        selected_collections = []

        if st.checkbox("바지 (pants)", value=True):
            selected_collections.append('pants')
        if st.checkbox("상의 (top)", value=True):
            selected_collections.append('top')
        if st.checkbox("아우터 (outer)", value=True):
            selected_collections.append('outer')
        if st.checkbox("원피스/스커트 (dress_skirts)", value=True):
            selected_collections.append('dress_skirts')

        if not selected_collections:
            st.warning("최소 1개 이상의 카테고리를 선택하세요")

        # 결과 수
        num_results = st.slider(
            "검색 결과 수",
            min_value=1,
            max_value=20,
            value=9,
            help="표시할 유사 아이템 개수"
        )

        # 탐지 옵션
        st.subheader("탐지 옵션")
        detection_threshold = st.slider(
            "탐지 신뢰도 임계값",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.1,
            help="낮을수록 더 많은 객체 탐지"
        )

    # 파일 업로더
    uploaded_file = st.file_uploader(
        "패션 이미지 업로드",
        type=['png', 'jpg', 'jpeg'],
        help="의류가 포함된 이미지를 업로드하세요"
    )

    if uploaded_file is not None:
        # 이미지 로드
        image = Image.open(uploaded_file).convert('RGB')

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("업로드된 이미지")
            st.image(image, width="stretch")

        # 의류 감지
        with st.spinner("의류 영역 감지 중..."):
            detected_items = detect_fashion_items(
                image,
                threshold=detection_threshold
            )

        if detected_items:
            # 정상 탐지 케이스: 후보를 보여주고 사용자가 직접 아이템을 선택한다
            # (기존 수동 선택 UX 유지).
            with col2:
                st.subheader(f"감지된 의류 아이템 ({len(detected_items)}개)")

                # 감지된 영역 미리보기
                preview_cols = st.columns(min(len(detected_items), 3))
                for idx, (item, preview_col) in enumerate(zip(detected_items[:3], preview_cols)):
                    bbox = item['bbox']
                    cropped = crop_image(image, bbox)
                    with preview_col:
                        st.image(cropped, width="stretch")
                        st.caption(f"{item['label']} ({item['score']:.2f})")

            # 아이템 선택
            st.write("---")

            col_select, col_search = st.columns([2, 1])

            with col_select:
                selected_idx = st.selectbox(
                    "검색할 아이템 선택:",
                    range(len(detected_items)),
                    format_func=lambda x: f"아이템 {x + 1} - {detected_items[x]['label']} (신뢰도: {detected_items[x]['score']:.2f})"
                )

            with col_search:
                st.write("")  # 공간 맞추기
                st.write("")
                search_button = st.button(
                    "🔍 유사 아이템 검색",
                    type="primary",
                    use_container_width=True
                )

            # 선택된 아이템 표시
            selected_item = detected_items[selected_idx]
            cropped_image = crop_image(image, selected_item['bbox'])

            st.subheader("선택된 아이템")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(cropped_image, width="stretch")
                st.caption(f"{selected_item['label']} (신뢰도: {selected_item['score']:.2f})")

        else:
            # 탐지된 아이템이 없는 케이스: 가짜 detection을 만들지 않고, 공유
            # fallback 메커니즘(raw fallback)을 통해 원본 이미지를 사용한다.
            # fallback_used/fallback_reason이 그대로 기록되어 failure analysis에서
            # 이 경로를 추적할 수 있다. 선택할 후보가 없으므로 미리보기/선택박스는
            # 표시하지 않는다.
            cropped_image, preprocessing_metadata = resolve_fallback_image(image, detected_items)
            logger.info(
                f"탐지된 아이템 없음 - fallback 적용: "
                f"mode={preprocessing_metadata['mode']}, "
                f"reason={preprocessing_metadata['fallback_reason']}"
            )

            with col2:
                st.subheader("감지된 의류 아이템 (0개)")
                st.warning("⚠️ 의류 아이템을 찾지 못했습니다. 원본 이미지로 검색합니다.")

            st.write("---")

            search_button = st.button(
                "🔍 유사 아이템 검색",
                type="primary",
                use_container_width=True
            )

            st.subheader("선택된 아이템")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(cropped_image, width="stretch")
                st.caption("원본 이미지 (탐지된 아이템 없음)")

        # 검색 실행
        if search_button:
            if not selected_collections:
                st.error("⚠️ 검색할 카테고리를 최소 1개 이상 선택하세요")
            else:
                with st.spinner("유사한 아이템 검색 중..."):
                    similar_items = search_similar_items(
                        cropped_image,
                        top_k=num_results,
                        selected_collections=selected_collections
                    )

                st.write("---")

                if similar_items:
                    show_similar_items(similar_items)

                    # 통계 표시
                    st.write("---")
                    st.write("### 📊 검색 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("검색된 아이템", len(similar_items))
                    with col2:
                        avg_similarity = sum(item['similarity_score'] for item in similar_items) / len(similar_items)
                        st.metric("평균 유사도", f"{avg_similarity:.1f}%")
                    with col3:
                        max_similarity = max(item['similarity_score'] for item in similar_items)
                        st.metric("최고 유사도", f"{max_similarity:.1f}%")
                else:
                    st.warning("유사한 아이템을 찾지 못했습니다.")

    else:
        st.info("👆 이미지를 업로드하여 검색을 시작하세요")

    # 하단 정보
    st.write("---")
    st.markdown("""
    **💡 팁:**
    - 의류가 명확하게 보이는 이미지를 사용하세요
    - 여러 의류가 있는 경우 원하는 아이템을 선택할 수 있습니다
    - 검색 결과는 유사도 순으로 정렬됩니다
    - 로그 파일(`musinsa_search.log`)에서 상세한 검색 결과를 확인할 수 있습니다
    """)

if __name__ == "__main__":
    main()
