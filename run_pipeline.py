"""전처리 실행: uv run run_pipeline.py"""
import argparse, logging, os, shutil, sys, time
import yaml
import kagglehub

sys.path.insert(0, "src")
from pipeline import load_raw, clean, make_labels, engineer, split_scale_save

def ensure_data(raw_path):
    """CSV가 없으면 kagglehub로 자동 다운로드"""
    if os.path.exists(raw_path):
        return
    print(f"  {raw_path} 없음 → Kaggle에서 다운로드 중...")
    downloaded = kagglehub.dataset_download("artermiloff/steam-games-dataset")
    # kagglehub는 캐시 디렉토리에 저장 → 거기서 CSV를 찾아 복사
    src = os.path.join(downloaded, "games_march2025_cleaned.csv")
    if not os.path.exists(src):
        # 파일명이 다를 수 있으니 csv 파일 탐색
        for f in os.listdir(downloaded):
            if f.endswith(".csv"):
                src = os.path.join(downloaded, f)
                break
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    shutil.copy2(src, raw_path)
    print(f"  ✅ 다운로드 완료: {raw_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("logs/pipeline.log", encoding="utf-8")])
    log = logging.getLogger("pipeline")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    meta = {"config_used": args.config}
    ensure_data(cfg["data"]["raw_path"])
    df = load_raw(cfg)
    df = clean(df, cfg, meta)
    df = make_labels(df, cfg, meta)
    df = engineer(df, cfg, meta)
    split_scale_save(df, cfg, meta)
    log.info(f"✅ 완료 ({time.time()-t0:.1f}초)")

if __name__ == "__main__":
    main()