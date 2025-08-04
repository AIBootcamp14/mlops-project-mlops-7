import torch
from model.model_arch import WideAndDeep
from model.preprocess import load_transaction_features, preprocess_for_inference

def run_inference_once(ml_session, checkpoint_path, logger):
    df = load_transaction_features(ml_session)
    X, y = preprocess_for_inference(df)

    # ✅ 모델 구조 학습 시와 동일하게 맞춰서 초기화
    model = WideAndDeep(
        wide_input_dim=3,
        deep_input_dim=14,
        num_classes=16,  # or load label_encoder
        hidden_dims=[128, 64],
        dropout_rate=0.1,
        use_softmax=False,
        batch_norm=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        wide_x = torch.tensor(X[0].values, dtype=torch.float32).to(device)
        deep_x = torch.tensor(X[1].values, dtype=torch.float32).to(device)
        outputs = model(wide_x, deep_x)

    # TODO: 후처리
    logger.info("Inference 완료, 결과 DB 저장 완료")