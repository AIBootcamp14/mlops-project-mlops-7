-- mlops database categories 테이블에 Dataset에 나온 데이터 기반 자료 넣기
-- name은 UNIQUE 처리가 되어있으며, name이면 무시함
use mlops; 

INSERT IGNORE INTO categories (name, gst_rate) VALUES
('Nest-USA', 0.10),
('Office', 0.10),
('Apparel', 0.18),
('Drinkware', 0.18),
('Lifestyle', 0.18),
-- ('Notebooks & Journals', 0.05),
('Headgear', 0.05),
('Waze', 0.18),
('Nest-Canada', 0.10),
('Bags', 0.18),
('Nest', 0.05),
('Bottles', 0.05),
('Gift Cards', 0.05),
('Housewares', 0.12),
('Android', 0.10),
('Accessories', 0.10);