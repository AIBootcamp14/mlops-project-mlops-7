# train.py
import os
import argparse
import datetime
import logging
import torch
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import dataset as module_data
import model as module_arch
import metric as module_metric
from utils import fix_seed, MetricTracker


def train(config, logger, tb_logger=None):
    fix_seed(config.seed)

    # Dataset
    train_dataset = getattr(module_data, config.dataset.type)(
        is_training=True,
        **config.dataset.args
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.args.batch_size,
        shuffle=config.dataloader.args.shuffle,
        num_workers=config.dataloader.args.num_workers,
        drop_last=True
    )

    # Input dimensions
    (x_wide, x_deep), _ = train_dataset[0]
    wide_input_dim = x_wide.shape[0]
    deep_input_dim = x_deep.shape[0]
    num_classes = int(train_dataset.y.max().item()) + 1

    # Model
    model_type = config.model.type
    if model_type == "WideAndDeep":
        # 명시적으로 필요한 인자만 꺼내고, 중복 방지
        model_args = dict(config.wideanddeep_args)
        model_args.pop("wide_input_dim", None)
        model_args.pop("deep_input_dim", None)
        model_args.pop("num_classes", None)

        model = module_arch.WideAndDeep(
            wide_input_dim=wide_input_dim,
            deep_input_dim=deep_input_dim,
            num_classes=num_classes,
            **model_args
        )
    elif model_type == "MLP":
        model = module_arch.MLP(
            input_size=wide_input_dim + deep_input_dim,
            output_size=num_classes,
            **config.mlp_args
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")

    if config.train.resume:
        model.load_state_dict(torch.load(config.train.resume_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = getattr(torch.nn, config.loss)().to(device)
    optimizer = getattr(torch.optim, config.optimizer.type)(
        model.parameters(), **config.optimizer.args)
    lr_scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.type)(
        optimizer, **config.lr_scheduler.args)

    metrics = [getattr(module_metric, met) for met in config.metrics]
    metric_tracker = MetricTracker('loss', *config.metrics)

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        metric_tracker.reset()

        for batch_idx, ((x_wide, x_deep), target) in enumerate(train_dataloader):
            x_wide, x_deep, target = x_wide.to(device), x_deep.to(device), target.to(device)

            if model_type == "MLP":
                x = torch.cat([x_wide, x_deep], dim=1)
                output = model(x)
            else:
                output = model(x_wide, x_deep)

            loss = criterion(output, target.view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_tracker.update('loss', loss.item())
            for met in metrics:
                pred = output.argmax(dim=1)
                metric_tracker.update(met.__name__, met(pred, target.view(-1).long()))

        lr_scheduler.step()

        logger.info(f"[Epoch {epoch}] " + ", ".join(
            [f"{k.upper()}: {v:.4f}" for k, v in metric_tracker.result().items()]))

        if epoch % config.train.save_period == 0:
            os.makedirs(config.train.save_dir, exist_ok=True)
            ckpt_path = os.path.join(config.train.save_dir, f"checkpoints/model-e{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")


def inference(config, checkpoint_path, output_path="inference_result.csv"):
    # 데이터셋
    dataset = getattr(module_data, config.dataset.type)(
        is_training=False,
        **config.dataset.args
    )
    (x_wide, x_deep), y = dataset[:][0], dataset[:][1]

    wide_input_dim = x_wide.shape[1]
    deep_input_dim = x_deep.shape[1]

    encoder = joblib.load(os.path.join(config.dataset.args.data_dir, 'label_encoder.pkl'))
    num_classes = len(encoder.classes_)
    print(f"num_classes from LabelEncoder: {num_classes}")

    # 모델 로드
    model_args = dict(config.wideanddeep_args)
    model_args.pop("wide_input_dim", None)
    model_args.pop("deep_input_dim", None)
    model_args.pop("num_classes", None)

    model = module_arch.WideAndDeep(
        wide_input_dim=wide_input_dim,
        deep_input_dim=deep_input_dim,
        num_classes=num_classes,
        **model_args
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일이 존재하지 않습니다: {checkpoint_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x_wide = x_wide.to(device)
        x_deep = x_deep.to(device)
        output = model(x_wide, x_deep)
        pred = output.argmax(dim=1).cpu().numpy()
        y_true = y.numpy()

    pred_label = encoder.inverse_transform(pred.astype(int))
    true_label = encoder.inverse_transform(y_true.astype(int))

    # 저장
    result_df = pd.DataFrame({
        "SampleID": list(range(len(pred))),
        "Prediction": pred,
        "Prediction_Label": pred_label,
        "GroundTruth": y_true,
        "GroundTruth_Label": true_label
    })
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Inference 완료! 결과 저장: {output_path}")


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
    if args.mode == "inference":
        if not args.checkpoint:
            raise ValueError("inference 모드에서는 --checkpoint 경로를 지정해야 합니다.")
        inference(config=config, checkpoint_path=args.checkpoint, output_path=args.output_path)
    logger.info("작업 완료")
