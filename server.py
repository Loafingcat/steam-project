"""MLflow UI + Streamlit을 한 번에 띄우기 (학습 없이 서빙만)

실행: uv run serve.py
종료: Ctrl+C (둘 다 같이 꺼짐)
"""
import subprocess
import sys
import time
import os

def main():
    procs = []
    try:
        print("🚀 MLflow UI 시작 (http://localhost:5000)")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui",
             "--backend-store-uri", "sqlite:///mlflow.db",
             "--port", "5000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        time.sleep(1)

        print("🚀 Streamlit 시작 (http://localhost:8501)")
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.port", "8501"],
        ))

        # Streamlit 프로세스가 살아있는 동안 대기
        procs[-1].wait()

    except KeyboardInterrupt:
        print("\n⏹️ 종료 중...")
    finally:
        for p in procs:
            p.terminate()
        print("✅ MLflow + Streamlit 종료")

if __name__ == "__main__":
    main()