"""
[의미 검색] ML vs DL 정량 비교 평가 (개선판)

변경점:
  1. 정답을 단일 태그 → 태그 '그룹'으로 (Roguelike가 Rogue-like/Rogue-lite/
     Action Roguelike 등으로 쪼개져 있어서 단일 태그론 0%가 나왔음)
  2. 쿼리를 세 유형으로 분리:
     - word(영어 키워드): ML 유리 예상
     - natural(자연어 문장): DL 유리 예상
     - korean(한국어): DL만 가능
"""
import logging
import mlflow
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")
from search_engine import load_catalog, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("eval_search")

TEST_QUERIES = [
    ("horror scary", ["Horror"], "word"),
    ("puzzle", ["Puzzle"], "word"),
    ("roguelike", ["Rogue-like", "Rogue-lite", "Action Roguelike",
                   "Traditional Roguelike", "Roguelike Deckbuilder"], "word"),
    ("pixel graphics", ["Pixel Graphics"], "word"),
    ("open world", ["Open World"], "word"),
    ("a calm relaxing game to unwind after work",
     ["Relaxing", "Cozy", "Casual"], "natural"),
    ("something scary to play at night alone", ["Horror"], "natural"),
    ("challenging game I can die and retry many times",
     ["Rogue-like", "Rogue-lite", "Action Roguelike", "Difficult"], "natural"),
    ("emotional game with a deep touching story",
     ["Story Rich", "Atmospheric"], "natural"),
    ("fun game to play together with friends",
     ["Co-op", "Multiplayer", "Online Co-Op", "PvP"], "natural"),
    ("잔잔한 힐링 게임", ["Relaxing", "Cozy", "Casual"], "korean"),
    ("무서운 공포 게임", ["Horror"], "korean"),
    ("친구랑 같이 하는 협동 게임", ["Co-op", "Multiplayer", "Online Co-Op"], "korean"),
    ("감성적인 스토리 게임", ["Story Rich", "Atmospheric"], "korean"),
]
K = 20


def hit(tags, gold_group):
    return any(g in tags for g in gold_group)


def precision_recall(catalog, idx, gold_group, k):
    top = idx[:k]
    hits = sum(hit(catalog.iloc[i]["tag_list"], gold_group) for i in top)
    total = catalog["tag_list"].apply(lambda t: hit(t, gold_group)).sum()
    return hits / k, (hits / total if total else 0.0)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("steam-ml-vs-dl")
    catalog = load_catalog()
    log.info(f"평가: {len(catalog):,}개 게임, {len(TEST_QUERIES)}개 쿼리")

    rows = []
    for method in ["ml", "dl"]:
        for query, gold, qtype in TEST_QUERIES:
            try:
                idx, _ = search(query, method=method, top_k=K)
            except Exception as e:
                log.warning(f"  {method} 실패 ({query[:20]}): {e}")
                continue
            p, r = precision_recall(catalog, idx, gold, K)
            rows.append({"method": method.upper(), "qtype": qtype,
                         "query": query, "precision": round(p, 3),
                         "recall": round(r, 3)})

    df = pd.DataFrame(rows)
    if df.empty:
        log.error("결과 없음 — 인덱스를 먼저 구축하세요")
        return
    df.to_json("data/processed/comparison_search.json",
               orient="records", force_ascii=False, indent=2)

    for method in df["method"].unique():
        sub = df[df["method"] == method]
        with mlflow.start_run(run_name=f"search-{method.lower()}"):
            mlflow.log_params({"method": method.lower(),
                               "model_family": method, "task": "semantic_search"})
            mlflow.log_metric(f"precision_at_{K}", sub["precision"].mean())
            for qt in sub["qtype"].unique():
                mlflow.log_metric(f"precision_{qt}",
                                  sub[sub["qtype"] == qt]["precision"].mean())

    log.info("-" * 60)
    pivot = df.pivot_table(index="qtype", columns="method",
                           values="precision", aggfunc="mean").round(3)
    pivot = pivot.reindex(["word", "natural", "korean"])
    log.info("쿼리 유형별 평균 Precision@20:\n" + pivot.to_string())
    log.info("-" * 60)


if __name__ == "__main__":
    main()
 