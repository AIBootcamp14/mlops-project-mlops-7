# [file] main.py  
# [description] FastAPI 서버 내 백그라운드 스레드로 운영 DB와 ML DB 동기화 작업 주기 실행
import os
from fastapi import FastAPI
import threading
import time
import logging

from infra.db.database import MainSessionLocal, MLSessionLocal
from service.sync_service import sync_data_to_ml_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(title="🧠 추론용 FastAPI 서버")

SYNC_INTERVAL = int(os.getenv("SYNC_INTERVAL", "100"))  

def sync_worker():
    """
    무한 루프를 돌며
    1) 운영 DB 세션 및 ML DB 세션 생성
    2) 데이터 동기화 작업 실행
    3) 커밋 또는 예외 발생 시 롤백 처리
    4) 세션 종료 후 주기 대기
    """
    while True:
        logging.info("🔄 데이터 동기화 시작")
        main_session = None
        ml_session = None
        try:
            main_session = MainSessionLocal()
            ml_session = MLSessionLocal()

            sync_data_to_ml_data(main_session, ml_session)

            main_session.commit()
            ml_session.commit()

            logging.info("✅ 동기화 완료")

        except Exception as e:
            logging.error(f"❌ 동기화 중 오류 발생: {e}", exc_info=True)
            if main_session:
                try:
                    main_session.rollback()
                except Exception:
                    pass
            if ml_session:
                try:
                    ml_session.rollback()
                except Exception:
                    pass
        finally:
            if main_session:
                main_session.close()
            if ml_session:
                ml_session.close()

        time.sleep(SYNC_INTERVAL)


@app.on_event("startup")
def start_sync_worker():
    """
    FastAPI 서버 시작 시 동기화 워커 스레드 백그라운드 실행
    """
    thread = threading.Thread(target=sync_worker, daemon=True)
    thread.start()


@app.get("/")
async def root():
    return {"message": "🧠 추론용 FastAPI 서버가 정상 작동 중입니다."}