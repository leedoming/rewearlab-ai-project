"""One-off script: build a real ChromaDB catalog from real Musinsa product
photos, run E0 (RAW) and E1-E4 (bbox policies) for real against it, and pool
each query's top-5 candidates from every one of those five runs for a human
to label. See docs/evidence/milestone-10-pilot.md for the full writeup.

Source images live OUTSIDE this repo -- two separate, unrelated personal
projects. pants/top/outer come from
C:/Users/smn07/Desktop/glacier-project/itda (a Musinsa crawl), organized
as <root>/<source folder>/<korean category>/<product name>.jpg.
dress_skirts comes from a second project,
C:/Users/smn07/Desktop/glacier-project/itda-fashion-detect (the only real
source found with any dress/skirt photos at all -- see
docs/evidence/milestone-10-final-decision.md section 2). Not part of the
reusable evaluation/ package: this is a one-time data preparation +
experiment-running step for a specific real-data pilot.

Usage:
    cd main && python evaluation/pilot/run_pilot.py
Writes:
    evaluation/pilot/_local/catalog_db/      (ChromaDB, not committed)
    evaluation/pilot/_local/pooled_candidates.json  (for manual labeling)
    evaluation/dataset/queries/Q0XX.jpg      (copied query images, committed)
    evaluation/dataset/query_manifest.csv    (committed)
"""

import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import chromadb
from PIL import Image

from retrieval.category import get_filtered_collections
from retrieval.config import COLLECTION_NAMES, DETECTION_THRESHOLD, MIN_BBOX_AREA
from retrieval.detection import detect_fashion_items
from retrieval.embedding import embed_image
from retrieval.models import load_detection_model, load_embedding_model
from retrieval.preprocessing import preprocess_image
from retrieval.search import search_collections

ITDA_ROOT = Path(r"C:\Users\smn07\Desktop\glacier-project\itda")
# A second, unrelated personal project -- the only real source found with any
# dress/skirt photos at all (itda has none). See docs/evidence/milestone-10-final-decision.md.
ITDA_FASHION_DETECT_ROOT = Path(r"C:\Users\smn07\Desktop\glacier-project\itda-fashion-detect\images\이미지")
PILOT_DIR = Path(__file__).parent
LOCAL_DIR = PILOT_DIR / "_local"
DATASET_DIR = PILOT_DIR.parent / "dataset"

FOLDER_TO_COLLECTION = {
    ITDA_ROOT / "musinsa_pants_1000" / "청바지": "pants",
    ITDA_ROOT / "musinsa_pants_1000" / "바지": "pants",
    ITDA_ROOT / "musinsa_upper_2000" / "긴팔": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "니트": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "맨투맨": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "반팔": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "셔츠": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "나시": "top",
    ITDA_ROOT / "musinsa_upper_2000" / "가디건": "outer",
    ITDA_ROOT / "musinsa_upper_2000" / "후드티": "outer",
    ITDA_FASHION_DETECT_ROOT / "원피스": "dress_skirts",
    ITDA_FASHION_DETECT_ROOT / "치마": "dress_skirts",
}

# dress_skirts source folders contain multi-product collage banners mixed in
# with real single-product photos (see scan_dress_skirt_source.py's output,
# dress_skirt_scan.json) -- only filenames confirmed (by real detection,
# <=2 detections with a dress/bottom label present) to show a single
# dress/skirt are eligible for the catalog or golden set at all. Two
# borderline cases were dropped by hand despite passing that filter:
# a lingerie-adjacent listing and one the detector mislabeled "top".
DRESS_SKIRT_ALLOWLIST = {
    "SHEIN 쉬인 ICON 빅 사이즈 회색 스판 데님 원피스  152335.jpg",
    "아란 여자 체형커버 수영복 비치 원피스수영복 빅사이즈 와플 1.jpg",
    "플루메리아 러플캉캉 쉬폰 원피스.jpg",
    "휴양지원피스 롱 비치 여름 해변 원피스 여행 드레스 동남아 바캉스 해외 공주 미니.jpg",
    "라인댄스치마 공연복 라틴댄스복 스포츠댄스복 블랙 쉬폰 주름 미니 수술 숏 댄스 티류.jpg",
    "벨트세트 레밋트 스판 모직 주름 롱 스커트.jpg",
    "((15%쿠폰한정수량)) 시크라인 가을신상 단독특가빅사이즈원피스하객룩오피스룩쉬폰플레어(S~XXL).jpg",
    "2023년 가을겨울 신상 여성 햅번 조거 민소매 롱원피스 9484486 922.jpg",
    "런칭가 39900원 러브썸 텍스처 블록 플레어 원피스 1종 빅사이즈 MN241F0001OP.jpg",
    "뷔스티에 롱원피스 여성 잔꽃 쉬폰 플라워.jpg",
    "블루종 한정수량특가 아이스 카울넥 민소매 원피스_B2407OP117B_M.jpg",
    "모란 타탄체크 주름 테니스치마 티류으스.jpg",
    "큐빅스커트 라인댄스복 라틴 파노티.jpg",
}

MAX_CATALOG_PER_COLLECTION = 60
RANDOM_SEED = 20260918
POOL_TOP_K = 5

# (query_id, folder_path, filename, category, difficulty, scene_type, num_visible_items, background_complexity)
QUERY_SPECS = [
    ("Q001", ITDA_ROOT / "musinsa_pants_1000" / "청바지", "062 테이퍼드 핏 데님팬츠 블랙진.jpg", "pants", "hard", "person_wearing", 2, "low"),
    ("Q002", ITDA_ROOT / "musinsa_pants_1000" / "청바지", "507 와이드 데님 - 화이트.jpg", "pants", "easy", "clean_product", 1, "low"),
    ("Q003", ITDA_ROOT / "musinsa_upper_2000" / "니트", "ANGORA PULLOVER KNIT_CREAM.jpg", "top", "hard", "person_wearing", 2, "medium"),
    ("Q004", ITDA_ROOT / "musinsa_upper_2000" / "반팔", "소사이어티 티셔츠 베이지.jpg", "top", "easy", "clean_product", 1, "low"),
    ("Q005", ITDA_ROOT / "musinsa_upper_2000" / "맨투맨", "ATHLETE STRIPE LS - [IVORY].jpg", "top", "medium", "clean_product", 1, "low"),
    ("Q006", ITDA_ROOT / "musinsa_upper_2000" / "가디건", "데일리 소프트 가디건&풀오버 SET 4종(택2).jpg", "outer", "hard", "multi_item", 2, "high"),
    ("Q007", ITDA_ROOT / "musinsa_upper_2000" / "후드티", "2010 Sports Hoodie Grey.jpg", "outer", "medium", "clean_product", 1, "low"),
    ("Q008", ITDA_FASHION_DETECT_ROOT / "원피스", "휴양지원피스 롱 비치 여름 해변 원피스 여행 드레스 동남아 바캉스 해외 공주 미니.jpg", "dress_skirts", "medium", "complex_background", 1, "high"),
    ("Q009", ITDA_FASHION_DETECT_ROOT / "원피스", "2023년 가을겨울 신상 여성 햅번 조거 민소매 롱원피스 9484486 922.jpg", "dress_skirts", "easy", "clean_product", 1, "low"),
    ("Q010", ITDA_FASHION_DETECT_ROOT / "치마", "벨트세트 레밋트 스판 모직 주름 롱 스커트.jpg", "dress_skirts", "medium", "multi_item", 2, "low"),
    ("Q011", ITDA_FASHION_DETECT_ROOT / "치마", "모란 타탄체크 주름 테니스치마 티류으스.jpg", "dress_skirts", "hard", "person_wearing", 1, "medium"),
]

BBOX_POLICIES = ["highest_confidence", "largest", "category_confidence", "category_largest"]


def gather_catalog_paths(exclude_paths):
    exclude = {str(p.resolve()) for p in exclude_paths}
    by_collection = {name: [] for name in COLLECTION_NAMES}
    for folder, collection in FOLDER_TO_COLLECTION.items():
        is_dress_skirts_source = folder.parent == ITDA_FASHION_DETECT_ROOT
        for path in sorted(folder.iterdir()):
            if not path.is_file() or str(path.resolve()) in exclude:
                continue
            if is_dress_skirts_source and path.name not in DRESS_SKIRT_ALLOWLIST:
                continue
            by_collection[collection].append(path)
    for collection in by_collection:
        random.Random(RANDOM_SEED).shuffle(by_collection[collection])
    return {c: paths[:MAX_CATALOG_PER_COLLECTION] for c, paths in by_collection.items() if paths}


def build_catalog(client, catalog_paths, detection_cache, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device):
    manifest = []
    for collection, paths in catalog_paths.items():
        coll = client.get_or_create_collection(name=collection)
        ids, embeddings, metadatas = [], [], []
        for i, path in enumerate(paths):
            product_id = f"{collection}-{i:04d}"
            with Image.open(path) as raw_image:
                image = raw_image.convert("RGB")
                detections = detect_fashion_items(
                    image, image_processor=image_processor, model=det_model, device=det_device,
                    threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
                )
                detection_cache[str(path)] = detections
                cropped, _ = preprocess_image(
                    image, detections, policy="category_confidence", category=collection,
                    fallback_policy="raw",
                )
                embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)
            ids.append(product_id)
            embeddings.append(embedding.tolist())
            metadatas.append({"id": product_id, "product_id": product_id, "name": path.name})
            manifest.append({"product_id": product_id, "collection": collection, "source_path": str(path)})
        if ids:
            coll.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        print(f"  {collection}: {len(ids)} items ingested")
    return manifest


def run_query(query_image_path, category, image_processor, det_model, det_device, embed_model, preprocess_fn, embed_device, client, top_k=POOL_TOP_K):
    """Return {policy_or_'raw': [result_dicts]} for this query across E0 + E1-E4."""
    with Image.open(query_image_path) as raw_image:
        image = raw_image.convert("RGB")
        detections = detect_fashion_items(
            image, image_processor=image_processor, model=det_model, device=det_device,
            threshold=DETECTION_THRESHOLD, min_area=MIN_BBOX_AREA,
        )

        results_by_mode = {}

        raw_embedding = embed_image(image, embed_model, preprocess_fn, embed_device)
        results_by_mode["E0_raw"] = search_collections(
            client, COLLECTION_NAMES, query_embedding=raw_embedding, top_k=top_k
        )

        for policy in BBOX_POLICIES:
            cropped, _ = preprocess_image(
                image, detections, policy=policy, category=category, fallback_policy="raw",
            )
            embedding = embed_image(cropped, embed_model, preprocess_fn, embed_device)
            results_by_mode[policy] = search_collections(
                client, COLLECTION_NAMES, query_embedding=embedding, top_k=top_k
            )

    return results_by_mode


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "queries").mkdir(parents=True, exist_ok=True)

    query_paths = [s[1] / s[2] for s in QUERY_SPECS]
    for spec, path in zip(QUERY_SPECS, query_paths):
        if not path.is_file():
            raise FileNotFoundError(f"{spec[0]}: {path} not found")

    print("Loading models (this downloads/loads yainage90 + Marqo SigLIP)...")
    image_processor, det_model, det_device = load_detection_model()
    embed_model, preprocess_fn, embed_device = load_embedding_model()

    client = chromadb.PersistentClient(path=str(LOCAL_DIR / "catalog_db"))

    print("Gathering catalog image paths...")
    catalog_paths = gather_catalog_paths(exclude_paths=query_paths)
    for collection, paths in catalog_paths.items():
        print(f"  {collection}: {len(paths)} candidate images")

    print("Building catalog (this is the slow part)...")
    detection_cache = {}
    manifest = build_catalog(
        client, catalog_paths, detection_cache, image_processor, det_model, det_device,
        embed_model, preprocess_fn, embed_device,
    )
    (LOCAL_DIR / "catalog_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Catalog built: {len(manifest)} total items across {len(catalog_paths)} collections.")

    print("Running queries (E0 + E1-E4) and pooling top-{} candidates...".format(POOL_TOP_K))
    pooled = {}
    manifest_rows = []
    for spec, query_path in zip(QUERY_SPECS, query_paths):
        query_id, folder_path, filename, category, difficulty, scene_type, num_items, bg_complexity = spec
        print(f"  {query_id} ({category}, {filename})")
        results_by_mode = run_query(
            query_path, category, image_processor, det_model, det_device,
            embed_model, preprocess_fn, embed_device, client,
        )

        pool = {}
        for mode, results in results_by_mode.items():
            for r in results:
                pool.setdefault(r["product_id"], r)
        pooled[query_id] = {
            "category": category,
            "results_by_mode": {mode: [r["product_id"] for r in results] for mode, results in results_by_mode.items()},
            "pooled_candidates": list(pool.values()),
        }

        dest_filename = f"{query_id}.jpg"
        shutil.copyfile(query_path, DATASET_DIR / "queries" / dest_filename)
        manifest_rows.append({
            "query_id": query_id,
            "image_path": f"queries/{dest_filename}",
            "category": category,
            "difficulty": difficulty,
            "scene_type": scene_type,
            "split": "dev",
            "num_visible_items": num_items,
            "background_complexity": bg_complexity,
            "important_features": "",
        })

    (LOCAL_DIR / "pooled_candidates.json").write_text(
        json.dumps(pooled, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (DATASET_DIR / "query_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "query_id", "image_path", "category", "difficulty", "scene_type",
            "split", "num_visible_items", "background_complexity", "important_features",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nDone.")
    print(f"Pooled candidates for labeling: {LOCAL_DIR / 'pooled_candidates.json'}")
    print(f"Catalog manifest (source paths): {LOCAL_DIR / 'catalog_manifest.json'}")
    print(f"Query manifest written to: {DATASET_DIR / 'query_manifest.csv'}")
    print(f"Query images copied to: {DATASET_DIR / 'queries'}")


if __name__ == "__main__":
    main()
