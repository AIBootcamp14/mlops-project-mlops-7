-- mlops database 생성하기 [백엔드 연동용]
CREATE DATABASE IF NOT EXISTS mlops;
USE mlops;

-- DB 유저와 권한 자동 생성
CREATE USER IF NOT EXISTS 'backend_notone' @'%' IDENTIFIED BY 'notone0818';

GRANT ALL PRIVILEGES ON mlops.* TO 'backend_notone' @'%';

FLUSH PRIVILEGES;

-- 새로운 DB 생성
CREATE DATABASE ml_data;

-- 'root' 계정이 'mlops'와 'ml_data' 모두 접근하도록 권한 부여
GRANT ALL PRIVILEGES ON mlops.* TO 'root'@'%';
GRANT ALL PRIVILEGES ON ml_data.* TO 'root'@'%';

-- 변경 사항 적용
FLUSH PRIVILEGES;