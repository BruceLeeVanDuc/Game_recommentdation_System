"""
Script kiểm tra xem project đã sẵn sàng deploy chưa
"""

import os
from pathlib import Path

def check_file_size(filepath, max_mb=50):
    """Kiểm tra kích thước file"""
    if not filepath.exists():
        return False, 0, f"MISSING: {filepath.name}"
    
    size_mb = filepath.stat().st_size / (1024**2)
    if size_mb > max_mb:
        return False, size_mb, f"TOO LARGE: {filepath.name} ({size_mb:.2f}MB > {max_mb}MB)"
    return True, size_mb, f"OK: {filepath.name} ({size_mb:.2f}MB)"

def main():
    print("="*60)
    print("KIEM TRA SAN SANG DEPLOY")
    print("="*60)
    
    all_ok = True
    total_size = 0
    
    # 1. Kiểm tra các file bắt buộc
    print("\n[1] CAC FILE BAT BUOC:")
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]
    
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            print(f"   [OK] {filename}")
        else:
            print(f"   [MISSING] {filename}")
            all_ok = False
    
    # 2. Kiểm tra Streamlit config
    print("\n[2] STREAMLIT CONFIG:")
    config_path = Path(".streamlit/config.toml")
    if config_path.exists():
        print(f"   [OK] .streamlit/config.toml")
    else:
        print(f"   [WARNING] .streamlit/config.toml (optional)")
    
    # 3. Kiểm tra model files
    print("\n[3] MODEL FILES:")
    models_dir = Path("models")
    
    # Required model files
    required_models = [
        ("hybrid_similarity_lightweight.pkl", 15),  # max 15MB
        ("games_metadata_lightweight.pkl", 10),     # max 10MB
        ("tfidf_vectorizer.pkl", 5),                # max 5MB
        ("top_games_list.pkl", 5),                  # max 5MB
    ]
    
    for filename, max_size in required_models:
        filepath = models_dir / filename
        ok, size, msg = check_file_size(filepath, max_size)
        total_size += size
        
        if ok:
            print(f"   [OK] {msg}")
        else:
            print(f"   [ERROR] {msg}")
            all_ok = False
    
    # 4. Kiểm tra các file KHÔNG nên có (quá lớn)
    print("\n[4] FILE LON CAN LOAI BO (neu co):")
    should_ignore = [
        "models/hybrid_similarity.pkl",  # 122MB
        "models/df_content_limited.pkl",
    ]
    
    found_large = False
    for filepath in should_ignore:
        p = Path(filepath)
        if p.exists():
            size_mb = p.stat().st_size / (1024**2)
            print(f"   [WARNING] {filepath} ({size_mb:.2f}MB) - should be in .gitignore")
            found_large = True
    
    if not found_large:
        print(f"   [OK] No large files found")
    
    # 5. Kiểm tra data directory
    print("\n[5] DATA DIRECTORY:")
    data_dir = Path("data")
    if data_dir.exists():
        data_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
        data_size_mb = data_size / (1024**2)
        print(f"   [WARNING] data/ exists ({data_size_mb:.2f}MB) - should be in .gitignore")
        print(f"   [INFO] Data files are NOT needed for deploy!")
    else:
        print(f"   [OK] data/ not found (good for deploy)")
    
    # 6. Tổng kết
    print("\n" + "="*60)
    print("TONG KET:")
    print("="*60)
    print(f"Total model size: {total_size:.2f}MB")
    
    if total_size < 50:
        print(f"[OK] Total size < 50MB - Good for free tier!")
    elif total_size < 100:
        print(f"[WARNING] Total size {total_size:.2f}MB - May be slow on free tier")
    else:
        print(f"[ERROR] Total size {total_size:.2f}MB - Too large for free deployment!")
        all_ok = False
    
    print()
    if all_ok:
        print("[SUCCESS] Project san sang deploy!")
        print("\nNEXT STEPS:")
        print("1. git init (neu chua co)")
        print("2. git add .")
        print("3. git commit -m 'Ready for deploy'")
        print("4. Push len GitHub")
        print("5. Deploy tren Streamlit Cloud hoac Hugging Face")
        print("\nChi tiet: Xem file DEPLOY.md")
    else:
        print("[ERROR] Project chua san sang!")
        print("\nFIX:")
        print("- Chay: python create_lightweight_model.py --n_games 2000")
        print("- Dam bao cac file required da duoc tao")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

