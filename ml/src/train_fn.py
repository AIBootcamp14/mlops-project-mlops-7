import os
import torch
from torch.utils.data import DataLoader
from utils import fix_seed, MetricTracker
import dataset as module_data
import model as module_arch
import metric as module_metric

def train(config, logger):
    fix_seed(config.seed)

    train_dataset = getattr(module_data, config.dataset.type)(
        is_training=True, **config.dataset.args
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.args.batch_size,
        shuffle=config.dataloader.args.shuffle,
        num_workers=config.dataloader.args.num_workers,
        drop_last=True
    )

    (x_wide, x_deep), _ = train_dataset[0]
    wide_input_dim, deep_input_dim = x_wide.shape[0], x_deep.shape[0]
    num_classes = int(train_dataset.y.max().item()) + 1

    model_type = config.model.type
    if model_type == "WideAndDeep":
        model_args = dict(config.wideanddeep_args)
        model_args.update({
            "wide_input_dim": wide_input_dim,
            "deep_input_dim": deep_input_dim,
            "num_classes": num_classes
        })
        model = module_arch.WideAndDeep(**model_args)
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
    optimizer = getattr(torch.optim, config.optimizer.type)(model.parameters(), **config.optimizer.args)
    scheduler = getattr(torch.optim.lr_scheduler, config.lr_scheduler.type)(optimizer, **config.lr_scheduler.args)

    metrics = [getattr(module_metric, met) for met in config.metrics]
    tracker = MetricTracker('loss', *config.metrics)

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        tracker.reset()

        for batch_idx, ((x_wide, x_deep), target) in enumerate(train_dataloader):
            x_wide, x_deep, target = x_wide.to(device), x_deep.to(device), target.to(device)
            output = model(torch.cat([x_wide, x_deep], dim=1)) if model_type == "MLP" else model(x_wide, x_deep)
            loss = criterion(output, target.view(-1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tracker.update('loss', loss.item())
            for met in metrics:
                pred = output.argmax(dim=1)
                tracker.update(met.__name__, met(pred, target.view(-1).long()))

        scheduler.step()

        logger.info(f"[Epoch {epoch}] " + ", ".join(f"{k.upper()}: {v:.4f}" for k, v in tracker.result().items()))

        if epoch % config.train.save_period == 0:
            ckpt_path = os.path.join(config.train.save_dir, f"checkpoints/model-e{epoch}.pt")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")