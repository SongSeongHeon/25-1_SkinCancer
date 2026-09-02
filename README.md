# Skin Lesion Diagnosis Assistant System

본 프로젝트는 피부 병변 이미지를 이용한 다중 분류 기반 피부암 진단 보조 시스템 개발을 목표로 한다.

국내 피부암 환자는 지난 20년간 약 7배 증가하였으며, 흑색종(melanoma)은 치료 시기가 늦어질수록 사망 위험이 뚜렷하게 증가하는 것으로 보고된다. 그러나 피부암 진단은 여전히 피부과 전문의의 육안 검사에 크게 의존하고 있어, 진단을 보조할 수 있는 AI 시스템의 필요성이 제기된다. 본 프로젝트에서는 HAM10000 데이터셋을 기반으로 MobileNet 전이학습 모델을 학습하고, 이를 TensorFlow.js로 변환하여 서버 없이 브라우저에서 직접 추론하는 웹 기반 자동 진단 시스템으로 구현하였다.

- **기간**: 25-1학기 (설계 및 프로젝트 심화 I)
- **팀 구성**: 3조 — 금진호(21615005), 송성헌(21615019), 임종서(21615036), 조은채(22615043)
- **담당 교수**: 조용석, 강병익

## Role Distribution

| 팀원 | 역할 |
|---|---|
| 금진호 | 데이터셋 수집 및 전처리 |
| 송성헌 | 웹 UI 개선 및 통합, Firebase 연동 |
| 임종서 | 모델 구성 및 학습, 비교군 성능 분석 |
| 조은채 | 테스트 및 시연 시스템 구축, 성능 분석 및 시각화 |

본 저장소는 팀원 송성헌이 담당한 웹 UI 통합 및 Firebase 연동 결과물을 중심으로 정리한 개인 기록용 저장소이다.

## Project Overview

```text
Image Upload or Camera Capture
→ Skin Filter Model Inference (Skin / Non-Skin)
→ Lesion Classification Model Inference
→ Top-3 Probability Result
→ Firebase Firestore / Realtime Database Log
```

## Main Features

- 사진 업로드 및 카메라 촬영 기반 이미지 입력
- 피부 이미지 여부 사전 판별 (오분류 방지용 필터 모델)
- 피부 병변 7종 분류 및 Top-3 확률 제시
- Firebase Firestore, Storage, Realtime Database 연동 결과 저장
- 진단 기록 히스토리 페이지 제공
- 브라우저 내 TensorFlow.js 추론 구조 (서버 추론 없음)

## Project Structure

```text
25-1_SkinCancer/
├── README.md
├── .gitignore
├── firebase.json
├── .firebaserc
│
├── public/
│   ├── index.html
│   ├── faq.html
│   ├── history.html
│   ├── config.example.js
│   ├── config.js
│   ├── css/
│   ├── assets/
│   └── final_model_kaggle_version/
│       ├── skin_filter_model/
│       └── lesion_model/
│
├── training/
│   ├── Final_SkinDisease.ipynb
│   └── skin_cancer_model.h5
│
└── legacy/
    └── jscript/
```

## Source Code Description

| Path | Description |
|---|---|
| `public/index.html` | 이미지 업로드, 모델 로드, 추론 및 결과 표시 로직을 포함한 메인 페이지 |
| `public/faq.html` | 앱 소개 및 사용 안내 페이지 |
| `public/history.html` | Firebase 기반 진단 기록 조회 페이지 |
| `public/config.example.js` | Firebase 설정 템플릿 |
| `public/config.js` | 실제 Firebase 프로젝트 키 (저장소에 포함하지 않음) |
| `public/final_model_kaggle_version/skin_filter_model/` | 피부 여부 판별용 TensorFlow.js 변환 모델 |
| `public/final_model_kaggle_version/lesion_model/` | 피부 병변 7종 분류용 TensorFlow.js 변환 모델 |
| `training/Final_SkinDisease.ipynb` | HAM10000 데이터 전처리, MobileNet 학습, 평가 전체 과정 |
| `training/skin_cancer_model.h5` | 학습된 Keras 모델 원본 |
| `legacy/jscript/` | 초기 버전 스크립트 (현재 미사용, 참고용 보관) |

## Dataset

본 프로젝트는 [HAM10000 (Human Against Machine with 10000 training images)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 데이터셋을 사용한다. HAM10000은 오스트리아 그라츠 이과대학에서 수집한 피부 병변 진단용 이미지 데이터셋으로, 총 10,015장의 이미지로 구성된다.

분류 대상 7개 클래스(MEL, NV, BCC, AKIEC, BKL, DF, VASC)는 ISIC 2018, 2019, 2020 Challenge에서 공식 다중 분류 과제로 채택된 표준 구성이다. 임상적으로 중요하고 실제 진단에서 자주 나타나는 병변이며, 딥러닝 학습 시 클래스 간 시각적 유사성과 다양성을 함께 확보할 수 있어 기존 연구와의 비교 및 재현성 확보에도 적합하다.

데이터셋 원본 이미지 및 메타데이터는 용량 문제로 GitHub 저장소에 포함하지 않는다. 모델을 재학습하려면 데이터셋을 직접 다운로드한 뒤 `training/Final_SkinDisease.ipynb` 내 경로 설정에 맞게 배치해야 한다.

## Model Pipeline

학습은 Keras 기반으로 진행하며, MobileNet(ImageNet pretrained)을 사용한다. MobileNet은 일반적인 합성곱 대비 연산량과 파라미터 수가 적어 모바일 및 브라우저 환경에 최적화되어 있고, 모델 크기와 속도를 조절해 성능과 속도 사이 균형을 맞출 수 있으며, 사전 학습된 가중치를 지원하여 소규모 데이터셋에서도 빠르게 성능을 확보할 수 있다는 점에서 선택하였다.

```text
HAM10000 Metadata Split (train / val)
→ Class-wise Image Directory Construction
→ Data Augmentation (Minority Class)
→ MobileNet Transfer Learning
→ Class Weight 기반 불균형 보정
→ Keras .h5 Model
→ TensorFlow.js Conversion
```

학습된 `.h5` 모델은 `tensorflowjs_converter`를 통해 `model.json` + 가중치 shard 형태로 변환하여 `public/final_model_kaggle_version/`에 배치한다.

### 참고 연구 성능 비교

동일 데이터셋(HAM10000, 7-class) 기준 관련 연구는 다음과 같다.

| 연구 | 모델 | 정확도 | 추가 지표 |
|---|---|---|---|
| Adebiyi et al. (2024) | ALBEF | 94.11% | AUCROC 0.9426 |
| Utsha Saha et al. (2024) | YOLOv8x-cls | 86.2% | Precision 82.1%, F1 77.0% |
| Islam Cahya Wicaksana et al. (2025) | YOLOv11x-cls | 84.72% | Recall 84.74%, F1 84.06% |
| Xiaoyi Liu et al. (2024) | Skin Net | 86.7% | AUC 0.96 |

## Why Top-3

단일 예측 결과만 제시할 경우 모델의 예측 불확실성을 사용자가 인지할 수 없다. 의료 AI에서는 Top-N 확률을 함께 제시하는 방식이 국제적으로 권장되며, 특히 Nevus와 Melanoma, BKL과 AKIEC처럼 임상적으로도 구분이 어려운 병변 간 시각적 유사성을 고려할 때 Top-1만 제시하면 오진 가능성이 발생할 수 있다.

관련 연구에 따르면 Top-1 정확도는 53.6%에 그치는 반면 Top-3 정확도는 75%로 크게 상승하며, 이는 일반인의 Top-3 진단 정확도(76%)와 유사한 수준이다. Top-5 정확도는 89%로 피부과 전문의의 Top-3 정확도(90%)와 유사하지만, 정보 과잉으로 실용성이 떨어진다는 지적이 있다. 종합적으로 Top-3는 전문의 수준에 근접한 정확도를 유지하면서도 정보 과부하를 방지할 수 있는 균형점으로 판단하여 채택하였다.

## Web Application

`public/`은 Firebase Hosting 기반 정적 웹 애플리케이션이다.

```text
Skin Filter Model Load
→ Lesion Classification Model Load
→ Image Input (Upload / Camera)
→ Tensor Preprocessing (Resize 224x224, Normalize)
→ Skin Filter Inference
→ Lesion Classification Inference (Skin으로 판정된 경우만)
→ Top-3 Result Display
→ Firestore / Realtime Database Log
```

두 모델 모두 브라우저에서 직접 추론하며, 이미지 원본은 서버로 전송되지 않고 클라이언트에서 처리된다. 단, 사용자가 결과 저장에 동의한 경우 Firebase Storage에 이미지가 업로드된다.

## Requirements

모델 학습 및 변환에 사용하는 주요 패키지는 다음과 같다.

```text
tensorflow
tensorflowjs
pandas
scikit-learn
matplotlib
seaborn
```

설치 명령어는 다음과 같다.

```bash
pip install tensorflow tensorflowjs pandas scikit-learn matplotlib seaborn
```

웹 애플리케이션은 별도 빌드 과정 없이 정적 파일로 구동된다. 로컬 실행 예시는 다음과 같다.

```bash
cp public/config.example.js public/config.js
cd public
python -m http.server 8080
```

Firebase Hosting 배포 예시는 다음과 같다.

```bash
firebase login
firebase deploy
```

## Repository Policy

다음 파일은 GitHub 저장소에 포함하지 않는다.

```text
public/config.js (Firebase 실제 프로젝트 키)
HAM10000 원본 이미지 및 메타데이터
.firebase/ 캐시 디렉토리
.ipynb_checkpoints/
```

## Development Status

현재 포함된 구현 범위는 다음과 같다.

- HAM10000 기반 데이터 전처리 및 train/val 분할
- MobileNet 전이학습 모델 구조
- Class Weight 기반 클래스 불균형 보정
- 학습 곡선 및 Confusion Matrix 평가
- Keras → TensorFlow.js 모델 변환
- 피부 여부 사전 필터 모델
- 병변 7종 분류 및 Top-3 확률 제시
- 이미지 업로드 및 카메라 촬영 입력
- Firebase Firestore, Storage, Realtime Database 연동
- 진단 기록 히스토리 페이지

## Expected Impact

- **조기 진단 가능**: 육안 검사의 한계 보완, 진단 지연 문제 감소
- **의료 접근성 향상**: 이미지 업로드만으로 진단 가능, 대기 시간 감소
- **진단 일관성 향상**: 표준화된 딥러닝 기반 알고리즘으로 일관된 결과 제공
- **자가 진단 보조**: 간편한 병변 상태 확인을 통한 심리적 안정감 제공

## Notes

본 프로젝트는 피부 병변 분류 모델의 학부 팀 프로젝트(설계 및 프로젝트 심화 I) 결과물이다.

분류 결과는 연구 및 학습 목적의 보조 지표이며, 실제 의료 진단 또는 치료 판단 목적으로 사용하지 않는다. 피부 병변이 의심되는 경우 반드시 전문의와 상담해야 한다.
