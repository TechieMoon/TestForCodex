# Transformer 예제

이 저장소는 PyTorch로 구현한 간단한 Transformer 모델 학습 예제를 제공합니다. 기본적인 구조를 이해하고 직접 실험해볼 수 있습니다.

## 설치 방법

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 사용 방법

아래 명령어로 모델을 학습할 수 있습니다.

```bash
python src/train.py --epochs 5
```

학습이 끝나면 `model.pth` 파일이 생성되며, 이를 이용해 예측을 수행할 수 있습니다.

## 라이선스

MIT 라이선스를 따릅니다.

