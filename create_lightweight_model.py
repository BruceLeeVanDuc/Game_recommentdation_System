"""
Script tạo model nhẹ cho deploy từ model đầy đủ
Giảm kích thước từ 122MB xuống ~5-10MB
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def create_lightweight_model(n_games=2000):
    """
    Tạo model nhẹ bằng cách:
    1. Lấy top N games phổ biến nhất
    2. Chỉ giữ similarity matrix cho N games đó
    3. Giảm precision xuống float16
    """
    
    models_dir = Path("models")
    
    print(f"[1/4] Loading full model...")
    # Load full model
    with open(models_dir / "hybrid_similarity.pkl", "rb") as f:
        full_matrix = pickle.load(f)
    
    df_games = pd.read_pickle(models_dir / "games_metadata.pkl")
    
    print(f"   Current size: {full_matrix.shape}")
    print(f"   Current dtype: {full_matrix.dtype}")
    
    # Load top games list nếu có
    try:
        with open(models_dir / "top_games_list.pkl", "rb") as f:
            top_games_list = pickle.load(f)
        print(f"   Found top_games_list with {len(top_games_list)} games")
    except:
        top_games_list = None
    
    print(f"\n[2/4] Selecting top {n_games} games...")
    
    # Chọn games dựa trên số lượng ratings
    if 'positive_ratings' in df_games.columns and 'negative_ratings' in df_games.columns:
        df_games['total_ratings'] = df_games['positive_ratings'] + df_games['negative_ratings']
        df_games['positive_ratio'] = df_games['positive_ratings'] / (df_games['total_ratings'] + 1)
        df_games['score'] = df_games['total_ratings'] * df_games['positive_ratio']
        
        # Sắp xếp theo score
        df_sorted = df_games.sort_values('score', ascending=False)
        top_indices = df_sorted.head(n_games).index.tolist()
    else:
        # Fallback: lấy n games đầu
        top_indices = list(range(min(n_games, len(df_games))))
    
    print(f"   Selected {len(top_indices)} games")
    
    print(f"\n[3/4] Creating lightweight similarity matrix...")
    # Lấy subset của matrix
    lightweight_matrix = full_matrix[np.ix_(top_indices, top_indices)]
    
    # Convert sang float16 để giảm size
    if lightweight_matrix.dtype != np.float16:
        lightweight_matrix = lightweight_matrix.astype(np.float16)
    
    print(f"   New size: {lightweight_matrix.shape}")
    print(f"   New dtype: {lightweight_matrix.dtype}")
    
    # Lấy metadata tương ứng
    df_lightweight = df_games.iloc[top_indices].reset_index(drop=True)
    
    print(f"\n[4/4] Saving lightweight model...")
    
    # Save lightweight matrix
    output_file = models_dir / "hybrid_similarity_lightweight.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(lightweight_matrix, f)
    
    # Save lightweight metadata
    df_lightweight.to_pickle(models_dir / "games_metadata_lightweight.pkl")
    
    # Tính toán kích thước file
    import os
    size_mb = os.path.getsize(output_file) / (1024**2)
    
    print(f"\n[SUCCESS] HOAN THANH!")
    print(f"   Similarity matrix: {output_file.name} ({size_mb:.2f} MB)")
    print(f"   Metadata: games_metadata_lightweight.pkl")
    print(f"   Giam tu {full_matrix.shape[0]} xuong {lightweight_matrix.shape[0]} games")
    print(f"   Giam kich thuoc: 122MB -> {size_mb:.2f}MB")
    
    # Hướng dẫn sử dụng
    print(f"\n[HOW TO USE]:")
    print(f"   Edit app.py line 76-79:")
    print(f"   ")
    print(f"   with open(f'{{base_dir}}/hybrid_similarity_lightweight.pkl', 'rb') as f:")
    print(f"       sim_matrix = pickle.load(f)")
    print(f"   df = pd.read_pickle(f'{{base_dir}}/games_metadata_lightweight.pkl')")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_games", type=int, default=2000, 
                       help="Số lượng games giữ lại (default: 2000)")
    args = parser.parse_args()
    
    create_lightweight_model(n_games=args.n_games)

