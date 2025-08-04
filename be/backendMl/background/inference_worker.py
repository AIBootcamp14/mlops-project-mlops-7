import logging
import time
from model.inference import inference
from infra.db.database import MLSessionLocal

INFERENCE_INTERVAL = 60 * 60 * 24  # 3분

def inference_worker():
    logging.info("🧠 추론 워커 시작")
    while True:
        ml_session = None
        try:
            ml_session = MLSessionLocal()

            checkpoint_path = "model/model_weight/model_v1.pt" # To Do. 나중에 경로 숨기기

            # inference 함수는 내부에서 모델 로드와 추론 수행
            inference(ml_session, checkpoint_path, logging.getLogger())

            logging.info("✅ 추론 작업 완료")

        except Exception as e:
            logging.error(f"❌ 추론 중 오류 발생: {e}", exc_info=True)

        finally:
            if ml_session:
                ml_session.close()

        time.sleep(INFERENCE_INTERVAL)
