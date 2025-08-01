# train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import datetime
import logging
from omegaconf import OmegaConf

from train_fn import train
from inference_fn import inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--mode", type=str, choices=["train", "inference"], default="train", help="실행 모드")
    parser.add_argument("--checkpoint", type=str, help="inference 시 사용할 모델 체크포인트 경로")
    parser.add_argument("--output_path", type=str, default="inference_result.csv", help="inference 결과 저장 파일 경로")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    os.makedirs(os.path.join(config.train.save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(config.train.save_dir, 'logs'), exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    log_path = os.path.join(config.train.save_dir, f"logs/{datetime.datetime.now():%y%m%d_%H%M%S}.log")
    logger.addHandler(logging.FileHandler(log_path))

    if args.mode == "train":
        train(config=config, logger=logger)
    elif args.mode == "inference":
        if not args.checkpoint:
            raise ValueError("inference 모드에서는 --checkpoint 경로를 지정해야 합니다.")
        inference(config=config, checkpoint_path=args.checkpoint, output_path=args.output_path)

    logger.info("작업 완료")
