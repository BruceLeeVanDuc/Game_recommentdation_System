# 🚀 HƯỚNG DẪN DEPLOY CHI TIẾT

## ✅ Chuẩn bị (Đã hoàn thành)

- [x] Tạo model lightweight (7.6MB thay vì 122MB)
- [x] Sửa app.py để dùng file nhỏ
- [x] Tạo .gitignore
- [x] Tạo .streamlit/config.toml
- [x] Kiểm tra requirements.txt

## 📋 Các file cần thiết để deploy

```
END/
├── app.py                                    # Main app
├── requirements.txt                          # Dependencies
├── .gitignore                               # Loại bỏ file không cần
├── .streamlit/config.toml                   # Streamlit config
├── README.md                                # Documentation
└── models/
    ├── hybrid_similarity_lightweight.pkl    # 7.63 MB ✅
    ├── games_metadata_lightweight.pkl       # 2.77 MB ✅
    ├── tfidf_vectorizer.pkl                 # 0.2 MB ✅
    └── top_games_list.pkl                   # < 1 MB ✅
```

**Tổng kích thước: ~11 MB** → OK cho deploy!

---

## 🎯 CÁCH 1: STREAMLIT CLOUD (KHUYÊN DÙNG - MIỄN PHÍ)

### Bước 1: Tạo GitHub Repository

```bash
# Khởi tạo git (nếu chưa có)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit - Ready for deploy"

# Tạo repo mới trên GitHub (https://github.com/new)
# Sau đó:
git remote add origin https://github.com/YOUR_USERNAME/steam-game-recommender.git
git branch -M main
git push -u origin main
```

### Bước 2: Deploy trên Streamlit Cloud

1. **Truy cập:** https://share.streamlit.io
2. **Đăng nhập** bằng GitHub
3. **New app** → Chọn repository vừa tạo
4. **Cấu hình:**
   - **Branch:** main
   - **Main file path:** app.py
   - **Python version:** 3.9 hoặc cao hơn
5. **Deploy!** (Chờ 3-5 phút)

### Bước 3: Kiểm tra

- App sẽ có URL: `https://YOUR_USERNAME-steam-game-recommender.streamlit.app`
- Test các tính năng:
  - ✅ Tìm kiếm game
  - ✅ Gợi ý game tương tự
  - ✅ Lọc theo thể loại
  - ✅ Hiển thị top games

---

## 🎯 CÁCH 2: HUGGING FACE SPACES (MIỄN PHÍ)

### Bước 1: Tạo Space

1. Truy cập: https://huggingface.co/spaces
2. Click **Create new Space**
3. Cấu hình:
   - **Space name:** steam-game-recommender
   - **License:** MIT
   - **Space SDK:** Streamlit
   - **Space hardware:** CPU basic (free)

### Bước 2: Upload files

```bash
# Clone space về
git clone https://huggingface.co/spaces/YOUR_USERNAME/steam-game-recommender
cd steam-game-recommender

# Copy files cần thiết
cp ../END/app.py .
cp ../END/requirements.txt .
cp -r ../END/models .
cp -r ../END/.streamlit .

# Push lên
git add .
git commit -m "Deploy app"
git push
```

### Bước 3: Tạo README.md cho Space

Hugging Face cần file này để hiển thị:

```bash
echo "---
title: Steam Game Recommender
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Steam Game Recommendation System
" > README_HF.md
```

---

## 🎯 CÁCH 3: RENDER.COM (FREE TIER)

### Bước 1: Tạo file cấu hình

Tạo file `render.yaml`:

```yaml
services:
  - type: web
    name: steam-game-recommender
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
```

### Bước 2: Deploy

1. Push code lên GitHub
2. Truy cập: https://render.com
3. **New** → **Web Service**
4. Connect repository
5. Chọn **Free** plan
6. Deploy!

---

## 🐳 CÁCH 4: DOCKER (Cho các platform khác)

### Tạo Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build và run:

```bash
# Build image
docker build -t steam-game-recommender .

# Run container
docker run -p 8501:8501 steam-game-recommender
```

---

## ❗ TROUBLESHOOTING

### Lỗi 1: "File not found: hybrid_similarity_lightweight.pkl"

**Nguyên nhân:** Chưa tạo file lightweight

**Giải pháp:**
```bash
python create_lightweight_model.py --n_games 2000
```

### Lỗi 2: "Memory limit exceeded"

**Nguyên nhân:** File model quá lớn

**Giải pháp:** Giảm số games:
```bash
python create_lightweight_model.py --n_games 1000
```

### Lỗi 3: "Module not found"

**Nguyên nhân:** Dependencies không đúng

**Giải pháp:**
```bash
pip install -r requirements.txt --upgrade
```

### Lỗi 4: App chậm khi khởi động

**Bình thường!** Lần đầu load ~10MB data sẽ mất 10-15 giây. Sau đó Streamlit sẽ cache lại.

---

## 📊 KIỂM TRA TRƯỚC KHI DEPLOY

```bash
# 1. Test app local
streamlit run app.py

# 2. Kiểm tra kích thước files
du -sh models/*

# 3. Verify dependencies
pip install -r requirements.txt

# 4. Check git status
git status
```

---

## 🎉 SAU KHI DEPLOY THÀNH CÔNG

1. **Cập nhật README.md** với link demo
2. **Test đầy đủ** các tính năng
3. **Share link** với giảng viên/bạn bè
4. **Monitor logs** để phát hiện lỗi

### Update README với link:

```bash
# Sửa dòng 18 trong README.md
👉 **[XEM DEMO TẠI ĐÂY](https://your-app-url.streamlit.app)**
```

---

## 📈 NÂNG CẤP (Optional)

- [ ] Custom domain
- [ ] Analytics tracking
- [ ] User authentication
- [ ] Database integration
- [ ] CDN cho assets

---

## 💡 MẸO HAY

1. **Streamlit Cloud restart app:** Settings → Reboot app
2. **View logs:** Click "Manage app" → Logs
3. **Update app:** Chỉ cần git push, tự động deploy lại
4. **Private app:** Settings → Change to private

---

## 📞 HỖ TRỢ

Nếu gặp vấn đề:
1. Check logs trên platform
2. Test lại local: `streamlit run app.py`
3. Verify file sizes: `ls -lh models/`
4. Check Python version: `python --version`

---

🎮 **CHÚC BẠN DEPLOY THÀNH CÔNG!** 🚀

