# 🩺 피부암 진단 예측 시스템 (Skin Cancer Diagnosis)

> HAM10000 데이터셋 기반 MobileNet 전이학습으로 피부 병변 7종을 분류하고,
> Top-3 확률을 제시하는 브라우저 기반 진단 웹앱입니다.

- **기간**: 25-1학기
- **역할**: 개인 프로젝트
- **데이터셋**: [HAM10000 (Human Against Machine with 10000 training images)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **모델**: MobileNet 기반 전이학습 (Keras `.h5` → TensorFlow.js 변환)
- **배포**: Firebase Hosting (정적 웹앱, 브라우저 내 추론)

<br>

## 📋 목차

- [문제의식](#-문제의식)
- [시스템 아키텍처](#-시스템-아키텍처)
- [주요 기능](#-주요-기능)
- [폴더 구조](#-폴더-구조)
- [실행 방법](#-실행-방법)
- [모델 성능](#-모델-성능)
- [스크린샷](#-스크린샷)
- [향후 과제](#-향후-과제)

<br>

## 🔍 문제의식

피부 병변(모반, 흑색종 등)은 육안으로 양성/악성 구분이 어렵고, 조기 진단이 예후에 큰 영향을 미칩니다.
이 프로젝트는 **누구나 스마트폰 카메라로 찍은 사진 한 장으로 1차 스크리닝**을 받아볼 수 있는
경량 웹 진단 도구를 목표로 합니다.

<br>

## 🏗 시스템 아키텍처

```
[사용자 이미지 업로드/촬영]
          │
          ▼
┌─────────────────────┐
│  1차 필터 모델        │  → 입력 이미지가 "피부 이미지"인지 판별
│  (skin_filter_model) │     (오분류 방지를 위한 사전 게이트)
└─────────┬───────────┘
          │ 피부 이미지로 판정된 경우만 통과
          ▼
┌─────────────────────┐
│  병변 분류 모델        │  → 7개 클래스 확률 산출
│  (lesion_model)      │     (akiec/bcc/bkl/df/mel/nv/vasc)
└─────────┬───────────┘
          │
          ▼
   Top-3 확률 결과 표시
          │
          ▼
  Firebase Firestore / Realtime DB
   (이미지 + 예측 결과 로그 저장)
```

> 📌 두 모델 모두 브라우저에서 **TensorFlow.js**로 직접 추론합니다 (서버 추론 없음).
> 학습은 Python/Keras로 진행 후 `.h5` → `tfjs_converter`로 변환했습니다.

<br>

## ✨ 주요 기능

- 📁 사진 업로드 / 📷 카메라 촬영으로 이미지 입력
- 피부 이미지 여부 사전 필터링 (오탐 방지)
- 7개 피부 병변 클래스에 대한 Top-3 확률 제시
  - 광선각화증(akiec), 기저세포암(bcc), 양성각화증(bkl), 피부섬유종(df), 흑색종(mel), 모반(nv), 혈관병변(vasc)
- 진단 기록 히스토리 확인 (Firebase 연동)
- 완전 클라이언트 사이드 추론 (서버 비용 없음, 개인정보 이미지가 서버로 전송되지 않음)

<br>

## 📁 폴더 구조

```
25-1_SkinCancer/
├── public/                          # Firebase Hosting 배포 대상 (실행 가능한 웹앱)
│   ├── index.html                   # 메인 진단 페이지 (모델 로드 + 추론 로직 포함)
│   ├── faq.html / history.html      # 보조 페이지
│   ├── config.example.js            # Firebase 설정 템플릿 (커밋 O)
│   ├── config.js                    # 실제 Firebase 키 (커밋 X, .gitignore 처리)
│   ├── css/                         # 스타일시트
│   ├── assets/                      # 이미지 리소스
│   └── final_model_kaggle_version/  # TensorFlow.js 변환 모델
│       ├── skin_filter_model/       # 1차 피부 여부 필터 모델
│       └── lesion_model/            # 병변 7종 분류 모델
│
├── training/                        # 모델 학습 코드 및 원본 산출물
│   ├── Final_SkinDisease.ipynb      # 학습 노트북 (데이터 전처리 → MobileNet 학습 → 평가)
│   └── skin_cancer_model.h5         # 학습된 Keras 모델 원본
│
├── legacy/                          # 초기 버전 스크립트 (현재 미사용, 참고용 보관)
│   └── jscript/
│
├── firebase.json / .firebaserc      # Firebase Hosting 설정
├── .gitignore
└── README.md
```

<br>

## 🚀 실행 방법

### 웹앱만 로컬에서 실행 (모델 추론 확인용)

```bash
# public/config.example.js를 복사해 config.js 생성 후 본인 Firebase 프로젝트 값 입력
cp public/config.example.js public/config.js

cd public
python -m http.server 8080
# 브라우저에서 http://localhost:8080 접속
```

> Firebase 저장 기능(Firestore/Storage)을 쓰지 않고 **분류 결과만 확인**하려면
> `config.js`에 더미 값을 넣어도 모델 추론 자체는 정상 동작합니다.

### Firebase Hosting 배포

```bash
firebase login
firebase deploy
```

### 모델 재학습

`training/Final_SkinDisease.ipynb` 참고 (HAM10000 데이터셋 로컬 경로 설정 필요).

<br>

## 📊 모델 성능

| 항목 | 내용 |
|---|---|
| Backbone | MobileNet (ImageNet pretrained, top 레이어 교체) |
| 입력 크기 | 224 × 224 × 3 |
| 클래스 수 | 7 (akiec, bcc, bkl, df, mel, nv, vasc) |
| 평가지표 | Categorical Accuracy, Top-3 Categorical Accuracy |
| 클래스 불균형 대응 | Data Augmentation + Class Weight 적용 |

> 상세 학습 곡선/혼동행렬은 `training/Final_SkinDisease.ipynb` 참고

<br>

## 📸 스크린샷

<!-- TODO: 실제 스크린샷으로 교체
<p align="center">
  <img src="docs/screenshots/main_upload.png" width="45%" alt="메인 업로드 화면"/>
  <img src="docs/screenshots/result.png" width="45%" alt="진단 결과 화면"/>
</p>
-->

| 메인 화면 | 진단 결과 |
|---|---|
| _(스크린샷 추가 예정)_ | _(스크린샷 추가 예정)_ |

<br>

## 🔮 향후 과제

- [ ] 실제 임상 이미지(비 Kaggle 데이터) 대상 일반화 성능 검증
- [ ] Grad-CAM 등 설명가능성(XAI) 시각화 추가
- [ ] 모델 경량화(양자화)로 초기 로딩 속도 개선
- [ ] 다국어(영어) 지원

<br>

---

⚠️ **의료 면책 조항**: 본 시스템은 학술 프로젝트 목적의 프로토타입이며, 실제 의료 진단을 대체하지 않습니다. 피부 병변이 의심되는 경우 반드시 전문의와 상담하세요.
