"""전처리 실행: uv run run_pipeline.py"""
import argparse, logging, sys, time
import yaml

sys.path.insert(0, "src")
from pipeline import load_raw, clean, make_labels, engineer, split_scale_save

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
    df = load_raw(cfg)
    df = clean(df, cfg, meta)
    df = make_labels(df, cfg, meta)
    df = engineer(df, cfg, meta)
    split_scale_save(df, cfg, meta)
    log.info(f"✅ 완료 ({time.time()-t0:.1f}초)")

if __name__ == "__main__":
    main()