# e-commerce 데이터를 활용한 거래 데이터 기반 AI 개인 맞춤형 상품 추천 서비스

> 실제 e-commerce 서비스 (쿠팡 등)를 벤치마킹하여 클론 프로젝트 형식으로 진행한 개인화 추천 기반 쇼핑 서비스 구현 프로젝트
> 
- **프로젝트 기간:** 2025. 07. 28 ~ 2025. 08. 08

---

## **1. 서비스 구성 요소**

### **1.1 주요 기능**

- 기능 : 사용자별 카테고리 추천

### **1.2 사용자 흐름**

- 사용자 시나리오 :
    1. 웹사이트 접속
    2. 추천된 카테고리 아이템 확인

---

## **2. 활용 장비 및 협업 툴**

### **2.1 활용 장비**

- **개발 환경:** Ubuntu, Windows 11, Mac
- **테스트 장비:** *GPU RTX 3090*

### **2.2 협업 툴**

- **소스 관리:** GitHub
- **프로젝트 관리:** GitHub & Notion
- **커뮤니케이션:** Slack
- **버전 관리:** Git

---

## **3. 최종 선정 AI 모델 구조**

- **모델 이름:** Wide & Deep
- **구조 및 설명:** 기억 기반의 선형 모델(Wide)과 일반화 능력을 갖춘 딥 뉴럴 네트워크(Deep)를 결합해 추천 성능을 향상시키는 구조
- **학습 데이터:**  [이커머스 고객 세분화 분석 아이디어 경진대회 - DACON](https://dacon.io/competitions/official/236222/data)
- **평가 지표:**  *Accuracy, Precision, Recall, F1-score*

---

## **4. 서비스 아키텍처**

### **4.1 시스템 구조도**

![서비스 아키텍처 예시]([스크린샷(97).png](https://github.com/AIBootcamp14/mlops-project-mlops-7/blob/main/assets/스크린샷(97).png))

### **4.2 데이터 흐름도**

1. 사용자 거래 데이터 생성 → 1H 단위로 inference용 데이터 ETL  → 1D 마다  추론 → 결과 반환
2. 일정 시간 기준 re-train → 재학습된 .pt 반환

---

## **5. 사용 기술 스택**

![기술 스택]([7조.jpg](https://github.com/AIBootcamp14/mlops-project-mlops-7/blob/main/assets/7조.jpg))

---

## **6. 팀원 소개**

| 이름 | 역할 | 담당 기능 |
| --- | --- | --- |
| 안희원 | 팀장/백엔드 개발자 | 서버 구축, DB 관리, API 개발 및 연동, 배포 관리 |
| 민병호 | 프론트엔드 개발자 | UI/UX 디자인, 프론트엔드 개발 |
| 김명철 | AI 모델 개발자 | AI 모델 선정 및 학습, 데이터 분석 |
| 정민지 | 데이터 엔지니어 | 데이터 수집, 전처리, 성능 평가 및 테스트 |

---

## **7. Appendix**

### **7.1 설치 및 실행 방법**

1. **필수 라이브러리 설치:**
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
2. **서버 실행:**
    
    ```bash
    # be/로 접속 시
    docker compose up 
    
    # k8s로 접속 시 각 폴더 내부로 가서
    kubectl apply f .
    # 일부 필요한 SQL 문은 순차적으로 실행
    ```
    
3. **웹페이지 접속:**
    
    ```
    <http://localhost:3000> # React
    <http://localhost:8080> # Airflow
    <http://localhost:5000> # Mlflow
    <http://localhost:3306> # mysql
    <http://localhost:8000> # fastapi
    ```
