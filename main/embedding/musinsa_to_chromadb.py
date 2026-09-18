"""
무신사 크롤링 데이터를 ChromaDB에 임베딩
객체 탐지(Object Detection)로 상품만 크롭 후 임베딩
카테고리별 라벨 매핑 적용

Detection / bbox selection / crop / embedding now delegate to the shared
retrieval core (main/retrieval/) so this file only handles data loading,
Streamlit UI and ChromaDB writes.
"""

import streamlit as st
import chromadb
import logging
import os
import sys
from PIL import Image
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import shutil
import json
import requests
from io import BytesIO
from pathlib import Path
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.category import get_allowed_labels, get_category_from_filename, get_collection_name
from retrieval.config import (
    CATEGORY_LABEL_MAPPING,
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('musinsa_embedding_crop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def reset_database():
    """데이터베이스 초기화"""
    try:
        shutil.rmtree("./musinsa_fashion_db_crop", ignore_errors=True)
        logger.info("데이터베이스가 초기화되었습니다.")
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")

def load_clip_model():
    """CLIP 모델 로드"""
    try:
        logger.info("CLIP 모델 로딩 중...")
        model, preprocess_val, device = core_load_embedding_model(EMBEDDING_MODEL)
        logger.info(f"CLIP 모델이 {device}에 로드되었습니다.")
        return model, preprocess_val, device
    except Exception as e:
        logger.error(f"CLIP 모델 로드 중 오류 발생: {str(e)}")
        raise

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

def load_json_data(json_path):
    """JSON 파일에서 무신사 데이터 로드"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON 파일 로드 완료: {len(data)}개 상품")
        return data
    except Exception as e:
        logger.error(f"JSON 로드 중 오류 발생: {str(e)}")
        return []

def get_category_name(json_path):
    """JSON 파일명에서 카테고리 추출 (예: musinsa_바지.json -> '바지')"""
    return get_category_from_filename(json_path)

def download_image_from_url(image_url, timeout=10):
    """URL에서 이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=timeout)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        logger.error(f"이미지 다운로드 실패 ({image_url}): {str(e)}")
        return None

def detect_and_crop_fashion_items(image, category, image_processor, detection_model, device, min_size=MIN_BBOX_AREA):
    """
    패션 아이템 탐지 및 크롭 (카테고리별 라벨 매핑)

    Detection -> category_confidence bbox selection -> crop은 retrieval
    core(preprocessing.preprocess_image)에 위임한다. 카테고리에 매핑된 라벨이
    아예 없는 경우에는 기존과 동일하게 탐지 자체를 생략하고 None을 반환한다.

    Args:
        image: PIL Image
        category: 카테고리 (무신사 카테고리)
        image_processor: 탐지 모델 프로세서
        detection_model: 탐지 모델
        device: torch device
        min_size: 최소 크기 (픽셀, area 기준)

    Returns:
        {'image': cropped, 'label': str, 'score': float} 또는 None
        (None이면 호출자가 원본 이미지를 사용한다)
    """
    allowed_labels = get_allowed_labels(category)

    if not allowed_labels:
        logger.warning(f"카테고리 '{category}'에 매핑된 라벨이 없습니다. 원본 사용.")
        return None

    try:
        detections = core_detect_fashion_items(
            image,
            image_processor=image_processor,
            model=detection_model,
            device=device,
            threshold=DETECTION_THRESHOLD,
            min_area=min_size,
        )

        cropped_image, metadata = preprocess_image(
            image,
            detections,
            policy="category_confidence",
            category=category,
            fallback_policy="raw",
        )

        if metadata["fallback_used"]:
            return None

        return {
            "image": cropped_image,
            "label": metadata["selected_label"],
            "score": metadata["detection_score"],
        }

    except Exception as e:
        logger.error(f"객체 탐지 중 오류 발생: {str(e)}")
        return None

def save_cropped_image(image, save_dir, product_id):
    """크롭된 이미지를 로컬에 저장"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{product_id}.jpg")
        image.save(save_path, 'JPEG')
        return save_path
    except Exception as e:
        logger.error(f"이미지 저장 실패: {str(e)}")
        return None

def create_metadata(product_data, label='original'):
    """무신사 상품 데이터에서 메타데이터 생성"""
    try:
        product_id = str(product_data.get('상품ID', ''))

        return {
            'id': product_id,
            'product_id': product_id,
            'category': product_data.get('카테고리', ''),
            'brand': product_data.get('브랜드', ''),
            'name': product_data.get('제품명', ''),
            'price': str(product_data.get('가격', '')),
            'discount_rate': str(product_data.get('할인율', '0')),
            'product_url': product_data.get('상품URL', ''),
            'image_url': product_data.get('이미지URL', ''),
            'detected_label': label
        }
    except Exception as e:
        logger.error(f"메타데이터 생성 중 오류 발생: {str(e)}")
        return None

def process_products(
    products,
    clip_model,
    preprocess_val,
    clip_device,
    detection_processor,
    detection_model,
    detection_device,
    save_cropped=True,
    crop_save_dir='./musinsa_cropped_images',
    progress_placeholder=None
):
    """
    상품 배치 처리: 다운로드 -> 객체 탐지 -> 크롭 -> 임베딩
    """
    processed_items = []

    for product in products:
        product_id = product.get('상품ID', '')
        image_url = product.get('이미지URL', '')
        category = product.get('카테고리', '상의')

        if not product_id or not image_url:
            continue

        # 이미지 다운로드
        image = download_image_from_url(image_url)
        if image is None:
            continue

        try:
            # 객체 탐지 및 크롭
            detected_item = detect_and_crop_fashion_items(
                image,
                category,
                detection_processor,
                detection_model,
                detection_device
            )

            # 탐지된 객체가 없으면 원본 이미지 사용
            if detected_item is None:
                logger.info(f"상품 {product_id} ({category}): 매칭되는 객체 탐지 안됨, 원본 사용")
                cropped_image = image
                label = 'original'
                score = 0.0
            else:
                cropped_image = detected_item['image']
                label = detected_item['label']
                score = detected_item['score']

            # 크롭 이미지 저장 (선택적)
            local_path = None
            if save_cropped:
                local_path = save_cropped_image(
                    cropped_image,
                    crop_save_dir,
                    product_id
                )

            # 임베딩 생성
            features = embed_image(cropped_image, clip_model, preprocess_val, clip_device)

            # 메타데이터 생성
            metadata = create_metadata(product, label)
            if metadata is None:
                continue

            processed_items.append({
                'id': metadata['id'],
                'embedding': features.tolist(),
                'metadata': metadata,
                'uri': local_path if local_path else image_url
            })

            # Streamlit 진행 상황 표시
            if progress_placeholder:
                with progress_placeholder.container():
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.image(image, caption="원본", use_container_width=True)
                    with col2:
                        st.image(cropped_image, caption=f"크롭 ({label})", use_container_width=True)
                    with col3:
                        st.write(f"**브랜드**: {metadata['brand']}")
                        st.write(f"**제품명**: {metadata['name'][:50]}")
                        st.write(f"**카테고리**: {category}")
                        st.write(f"**탐지 라벨**: {label} (신뢰도: {score:.2f})")
                        st.write(f"**가격**: {metadata['price']}원")

            # API 과부하 방지
            time.sleep(0.1)

        except Exception as e:
            logger.error(f"상품 처리 중 오류 ({product_id}): {str(e)}")
            continue

    return processed_items

def add_to_collection(collection, processed_items):
    """배치 단위로 컬렉션에 추가"""
    if not processed_items:
        return 0

    try:
        collection.add(
            ids=[item['id'] for item in processed_items],
            embeddings=[item['embedding'] for item in processed_items],
            metadatas=[item['metadata'] for item in processed_items],
            uris=[item['uri'] for item in processed_items]
        )
        return len(processed_items)
    except Exception as e:
        logger.error(f"컬렉션 추가 중 오류 발생: {str(e)}")
        return 0

def main():
    st.set_page_config(layout="wide")
    st.title("🛍️ 무신사 상품 객체 탐지 + 임베딩 (카테고리별)")

    st.markdown("""
    ### 사용 방법
    1. 카테고리별 JSON 파일 경로를 입력 (파일명에 카테고리 포함 필수)
       - 예: `musinsa_바지.json`, `musinsa_상의.json`, `musinsa_아우터.json`, `musinsa_원피스_스커트.json`
    2. 각 카테고리는 자동으로 해당 컬렉션에 임베딩
       - **바지** → `pants` 컬렉션
       - **상의** → `top` 컬렉션
       - **아우터** → `outer` 컬렉션
       - **원피스_스커트** → `dress_skirts` 컬렉션
    3. 탐지된 객체는 크롭하여 임베딩, 탐지 안되면 원본 이미지로 임베딩
    """)

    # JSON 파일 경로들 입력
    st.write("### 📁 JSON 파일 경로 입력")

    json_files = []

    col1, col2 = st.columns(2)

    with col1:
        pants_path = st.text_input(
            "바지 (pants)",
            value="musinsa_바지.json",
            help="바지 카테고리 JSON 파일"
        )
        if pants_path and os.path.exists(pants_path):
            json_files.append(pants_path)
            st.success(f"✓ {pants_path}")
        elif pants_path:
            st.warning(f"⚠ 파일 없음: {pants_path}")

        outer_path = st.text_input(
            "아우터 (outer)",
            value="musinsa_아우터.json",
            help="아우터 카테고리 JSON 파일"
        )
        if outer_path and os.path.exists(outer_path):
            json_files.append(outer_path)
            st.success(f"✓ {outer_path}")
        elif outer_path:
            st.warning(f"⚠ 파일 없음: {outer_path}")

    with col2:
        top_path = st.text_input(
            "상의 (top)",
            value="musinsa_상의.json",
            help="상의 카테고리 JSON 파일"
        )
        if top_path and os.path.exists(top_path):
            json_files.append(top_path)
            st.success(f"✓ {top_path}")
        elif top_path:
            st.warning(f"⚠ 파일 없음: {top_path}")

        dress_path = st.text_input(
            "원피스_스커트 (dress_skirts)",
            value="musinsa_원피스_스커트.json",
            help="원피스/스커트 카테고리 JSON 파일"
        )
        if dress_path and os.path.exists(dress_path):
            json_files.append(dress_path)
            st.success(f"✓ {dress_path}")
        elif dress_path:
            st.warning(f"⚠ 파일 없음: {dress_path}")

    if not json_files:
        st.error("처리할 JSON 파일이 없습니다. 파일 경로를 확인해주세요.")
        return

    st.info(f"총 {len(json_files)}개 카테고리 파일 발견")

    # 옵션 설정
    st.write("### ⚙️ 옵션 설정")
    col1, col2 = st.columns(2)

    with col1:
        save_cropped = st.checkbox(
            "크롭 이미지 저장",
            value=True,
            help="크롭된 이미지를 로컬에 저장"
        )

    with col2:
        batch_size = st.number_input(
            "배치 크기",
            min_value=1,
            max_value=20,
            value=5,
            help="한 번에 처리할 상품 수"
        )

    # DB 초기화
    if st.checkbox("⚠️ 데이터베이스 초기화"):
        if st.button("초기화 실행"):
            reset_database()
            st.success("데이터베이스 초기화 완료")

    # 임베딩 시작
    if st.button("🚀 카테고리별 객체 탐지 + 임베딩 시작", type="primary"):
        try:
            # 전체 진행 상황
            st.write("---")
            overall_status = st.empty()

            # 모델 로드 (한 번만)
            with st.spinner("모델 로딩 중..."):
                clip_model, preprocess_val, clip_device = load_clip_model()
                detection_processor, detection_model, detection_device = load_detection_model()

            # ChromaDB 클라이언트 초기화
            # 환경 변수로 ChromaDB 서버 설정
            import os
            chromadb_host = os.getenv("CHROMADB_HOST", None)
            chromadb_port = int(os.getenv("CHROMADB_PORT", 8000))

            if chromadb_host:
                # 원격 ChromaDB 서버 사용
                client = chromadb.HttpClient(host=chromadb_host, port=chromadb_port)
                st.info(f"ChromaDB 원격 서버 연결: {chromadb_host}:{chromadb_port}")
            else:
                # 로컬 파일 기반 ChromaDB 사용
                client = chromadb.PersistentClient(path="./musinsa_fashion_db_crop")
                st.info("ChromaDB 로컬 파일 사용: ./musinsa_fashion_db_crop")
            embedding_function = OpenCLIPEmbeddingFunction()
            data_loader = ImageLoader()

            # 전체 통계
            total_stats = {
                'total_files': len(json_files),
                'processed_files': 0,
                'total_products': 0,
                'total_added': 0,
                'total_original': 0
            }

            # 카테고리별 처리
            for file_idx, json_path in enumerate(json_files, 1):
                st.write("---")
                st.write(f"## 📦 {file_idx}/{len(json_files)} - {os.path.basename(json_path)}")

                # 카테고리 추출
                category = get_category_name(json_path)
                if category is None:
                    st.error(f"파일명에서 카테고리를 추출할 수 없습니다: {json_path}")
                    continue

                collection_name = get_collection_name(category)
                st.info(f"**카테고리**: {category} → **컬렉션**: `{collection_name}`")

                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                progress_placeholder = st.empty()

                # JSON 로드
                status_text.text(f"[{category}] JSON 데이터 로딩 중...")
                products = load_json_data(json_path)

                if not products:
                    st.warning(f"JSON 파일에 상품이 없습니다: {json_path}")
                    continue

                total_products = len(products)
                total_stats['total_products'] += total_products
                st.write(f"총 **{total_products}개** 상품 처리 시작")

                # 컬렉션 생성
                status_text.text(f"[{category}] 컬렉션 생성 중...")
                collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    data_loader=data_loader
                )

                # 카테고리별 통계
                category_stats = {
                    'total': total_products,
                    'processed': 0,
                    'added': 0,
                    'original': 0
                }

                # 배치 처리
                for i in range(0, total_products, batch_size):
                    batch_products = products[i:i + batch_size]

                    status_text.text(
                        f"[{category}] 처리 중... {i + 1}~{min(i + batch_size, total_products)}/{total_products}"
                    )

                    # 객체 탐지 + 크롭 + 임베딩
                    processed_items = process_products(
                        batch_products,
                        clip_model,
                        preprocess_val,
                        clip_device,
                        detection_processor,
                        detection_model,
                        detection_device,
                        save_cropped=save_cropped,
                        crop_save_dir=f'./musinsa_cropped_images/{collection_name}',
                        progress_placeholder=progress_placeholder
                    )

                    # ChromaDB 추가
                    added_count = add_to_collection(collection, processed_items)

                    # 원본 이미지 사용 횟수 계산
                    original_count = sum(1 for item in processed_items if item['metadata']['detected_label'] == 'original')

                    category_stats['processed'] += len(batch_products)
                    category_stats['added'] += added_count
                    category_stats['original'] += original_count

                    # 진행률
                    progress_bar.progress(category_stats['processed'] / category_stats['total'])

                # 카테고리별 결과
                st.success(f"✅ [{category}] 처리 완료!")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("전체 상품", category_stats['total'])
                with col2:
                    st.metric("처리 완료", category_stats['processed'])
                with col3:
                    st.metric("임베딩 성공", category_stats['added'])
                with col4:
                    st.metric("원본 사용", category_stats['original'])

                st.write(f"**컬렉션 총 임베딩 수**: {collection.count()}개")

                # 전체 통계 업데이트
                total_stats['processed_files'] += 1
                total_stats['total_added'] += category_stats['added']
                total_stats['total_original'] += category_stats['original']

                # 전체 진행 상황
                overall_status.write(
                    f"### 📊 전체 진행: {total_stats['processed_files']}/{total_stats['total_files']} 파일 완료"
                )

            # 최종 결과
            st.write("---")
            st.success("🎉 모든 카테고리 처리 완료!")

            st.write("### 📈 전체 통계")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("처리 파일", total_stats['processed_files'])
            with col2:
                st.metric("전체 상품", total_stats['total_products'])
            with col3:
                st.metric("총 임베딩", total_stats['total_added'])
            with col4:
                st.metric("원본 사용", total_stats['total_original'])

            st.write("### 📊 컬렉션 정보")
            for json_path in json_files:
                category = get_category_name(json_path)
                if category:
                    collection_name = get_collection_name(category)
                    try:
                        coll = client.get_collection(collection_name)
                        st.write(f"- **{category}** (`{collection_name}`): {coll.count()}개 임베딩")
                    except:
                        st.write(f"- **{category}** (`{collection_name}`): 컬렉션 없음")

            st.write(f"- DB 경로: `./musinsa_fashion_db_crop`")
            if save_cropped:
                st.write(f"- 크롭 이미지: `./musinsa_cropped_images/`")

            st.write("### 📋 카테고리별 라벨 매핑")
            for category, labels in CATEGORY_LABEL_MAPPING.items():
                collection_name = get_collection_name(category)
                st.write(f"- **{category}** (`{collection_name}`): {', '.join(labels)}")

        except Exception as e:
            st.error(f"❌ 오류: {str(e)}")
            logger.error(f"오류: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
