# Skin Lesion Diagnosis Web Application

본 프로젝트는 피부 병변 이미지를 이용한 브라우저 기반 피부암 진단 보조 시스템 개발을 목표로 한다.

피부 병변은 육안으로 양성과 악성을 구분하기 어렵고, 조기 진단 여부가 예후에 큰 영향을 미친다. 본 프로젝트에서는 HAM10000 데이터셋을 기반으로 MobileNet 전이학습 모델을 학습하고, 학습된 모델을 TensorFlow.js로 변환하여 서버 없이 브라우저에서 직접 추론하는 정적 웹 애플리케이션으로 구현하였다.

입력 이미지는 먼저 피부 여부 판별 모델을 통과한 뒤, 피부 이미지로 판정된 경우에만 병변 분류 모델을 거쳐 상위 확률 결과를 제시한다.

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

본 프로젝트는 [HAM10000 (Human Against Machine with 10000 training images)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) 데이터셋을 사용한다.

데이터셋 원본 이미지 및 메타데이터는 용량 문제로 GitHub 저장소에 포함하지 않는다. 모델을 재학습하려면 데이터셋을 직접 다운로드한 뒤 `training/Final_SkinDisease.ipynb` 내 경로 설정에 맞게 배치해야 한다.

분류 대상 클래스는 다음과 같다.

```text
akiec  광선각화증
bcc    기저세포암
bkl    양성각화증
df     피부섬유종
mel    흑색종
nv     모반
vasc   혈관병변
```

## Model Pipeline

학습은 Keras 기반으로 진행하며, MobileNet(ImageNet pretrained)의 상위 레이어를 교체하여 7-class 분류기로 재구성한다.

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

## Notes

본 프로젝트는 피부 병변 분류 모델의 학부 개인 프로젝트 결과물이다.

분류 결과는 학습 목적의 보조 지표이며, 실제 의료 진단 또는 치료 판단 목적으로 사용하지 않는다. 피부 병변이 의심되는 경우 반드시 전문의와 상담해야 한다.
