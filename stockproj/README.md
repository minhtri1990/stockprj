# Dự án stockproj

Dự án Streamlit sử dụng thư viện vnstock để lấy dữ liệu chứng khoán Việt Nam, hiển thị biểu đồ giá, báo cáo tài chính, so sánh nhiều mã và phân tích kỹ thuật (RSI, MACD, MA).

## 1. Cấu trúc thư mục

```
stockproj/
├─ README.md
├─ requirements.txt
├─ Procfile
├─ Dockerfile
├─ app.py
├─ utils/
│  ├─ __init__.py
│  ├─ data_service.py
│  ├─ charts.py
│  ├─ indicators.py
│  ├─ utils.py
│  └─ pages/
│     ├─ 1_Thong_Tin_Co_Phieu.py
│     ├─ 2_Lich_Su_Gia.py
│     ├─ 3_Bao_Cao_Tai_Chinh.py
│     ├─ 4_So_Sanh.py
│     └─ 5_Phan_Tich_Ky_Thuat.py
└─ .streamlit/
   └─ config.toml
```

(Để tránh lỗi hệ thống file, mình dùng tên file ASCII thay vì emoji; bạn có thể đổi lại.)

---

## 2. requirements.txt


## 3. start
venv\Scripts\activate
streamlit run app.py

```bash
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate (Windows)
venv\Scripts\activate
pip install -r requirements.txt
# pip install --force-reinstall -r requirements.txt
streamlit run app.py
```

## Docker

```bash
docker build -t stockproj .
docker run -p 8501:8501 stockproj
```

## Ghi chú
- API vnstock có thể thay đổi theo phiên bản.
- Interval 1W / 1M được tự resample nếu chưa hỗ trợ.
- Không dùng cho mục đích giao dịch thời gian thực.
