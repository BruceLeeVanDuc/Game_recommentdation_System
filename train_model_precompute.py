"""
TRAIN MODEL (PRECOMPUTE FULL SIMILARITY MATRIX) + BỔ SUNG DỮ LIỆU

Mục tiêu:
- Train và lưu `models/hybrid_similarity.pkl` dưới dạng MA TRẬN NxN (float16)
- Load ĐẦY ĐỦ dữ liệu bổ sung (media, description) giống train_model_safe.py
- Tương thích trực tiếp với `app.py`/`cli_app.py` (đang enumerate(sim_matrix[idx]))

Khuyến nghị:
- Đừng để N quá lớn (ví dụ > 12k) vì NxN sẽ rất nặng RAM/đĩa.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Config:
    data_dir: str = "data"
    output_dir: str = "models"
    max_features: int = 5000
    # weights
    w_collab: float = 0.60
    w_hours: float = 0.25
    w_content: float = 0.10
    w_sentiment: float = 0.05
    # runtime limits
    min_users_per_game: int = 5


def clean_text(text: object) -> str:
    return str(text).lower().replace(" ", "")


def convert_hours_to_rating(hours: float) -> float:
    """Chuyển đổi giờ chơi sang thang điểm 1-5 (rule-based)"""
    if hours < 2.0:
        return 1.0
    if hours < 10.0:
        return 2.0
    if hours < 50.0:
        return 3.0
    if hours < 100.0:
        return 4.0
    return 5.0


def pick_limit_by_ram() -> int:
    avail_gb = psutil.virtual_memory().available / (1024**3)
    if avail_gb >= 24:
        return 15000
    if avail_gb >= 16:
        return 12000
    return 8000


def _is_colab() -> bool:
    try:
        import google.colab  # type: ignore

        return True
    except Exception:
        return False


def _maybe_mount_drive(mount: bool) -> None:
    if not mount:
        return
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive", force_remount=False)
    except Exception:
        return


def _resolve_data_dir(data_dir: str) -> str:
    """
    Nếu không thấy các file dữ liệu trong `data_dir`, tự dò các vị trí phổ biến (Colab/Drive).
    Ưu tiên giống train_model_safe.py: /content/drive/MyDrive/KHDL_CVS/
    """
    base = Path(data_dir)
    required = ["steam.csv", "steamspy_tag_data.csv", "steam_requirements_data.csv", "steam-200k.csv"]

    def ok(p: Path) -> bool:
        return all((p / f).exists() for f in required)

    # 1) data_dir hiện tại
    if ok(base):
        return str(base)

    # 2) nếu user truyền "data" nhưng thực tế là root repo -> thử ../data theo cwd
    if base.name == "data":
        if ok(Path(".") / "data"):
            return str(Path(".") / "data")

    # 3) các candidate phổ biến
    candidates = [
        Path("data"),
        Path("/content/data"),
        Path("/content/drive/MyDrive/KHDL_CVS"),
        Path("/content/drive/MyDrive/KHDL_CVS/data"),
        Path("/content/drive/MyDrive"),
    ]
    for c in candidates:
        if ok(c):
            return str(c)

    # 4) scan bounded in Drive (match folder chứa steam.csv)
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        try:
            for p in drive_root.rglob("steam.csv"):
                parent = p.parent
                if ok(parent):
                    return str(parent)
        except Exception:
            pass

    return str(base)  # fallback: để fail rõ ràng


def load_and_process_data(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DỮ LIỆU ĐẦY ĐỦ: core + bổ sung (media, description) giống train_model_safe.py
    """
    data_dir = Path(cfg.data_dir)
    print(f"[1/4] Loading data from: {data_dir}")

    # --- CORE FILES ---
    df_games = pd.read_csv(data_dir / "steam.csv")
    df_tags = pd.read_csv(data_dir / "steamspy_tag_data.csv")
    df_reqs = pd.read_csv(data_dir / "steam_requirements_data.csv")
    df_users = pd.read_csv(
        data_dir / "steam-200k.csv",
        header=None,
        names=["UserID", "Game", "Action", "Hours", "Unused"],
    )
    print("   OK - Loaded core CSV files")

    # --- BỔ SUNG (OPTIONAL) ---
    df_media: Optional[pd.DataFrame] = None
    df_desc: Optional[pd.DataFrame] = None

    try:
        df_media = pd.read_csv(data_dir / "steam_media_data.csv")
        print("   OK - Loaded steam_media_data.csv")
    except FileNotFoundError:
        print("   WARN - steam_media_data.csv not found (skipped)")

    try:
        df_desc = pd.read_csv(data_dir / "games_march2025_cleaned.csv")
        print("   OK - Loaded games_march2025_cleaned.csv")
    except FileNotFoundError:
        print("   WARN - games_march2025_cleaned.csv not found (skipped)")

    # --- XỬ LÝ REQUIREMENTS ---
    df_reqs.columns = df_reqs.columns.str.strip().str.lower()
    if "steam_appid" in df_reqs.columns:
        df_reqs = df_reqs.rename(columns={"steam_appid": "appid"})

    # --- XỬ LÝ TAGS ---
    tag_columns = df_tags.columns[1:]
    df_tags["all_tags"] = df_tags.apply(lambda row: " ".join([tag for tag in tag_columns if row[tag] > 0]), axis=1)
    # Tạo steamspy_tags dict (cho app.py hiển thị)
    df_tags["steamspy_tags"] = df_tags.apply(lambda row: {tag: row[tag] for tag in tag_columns if row[tag] > 0}, axis=1)

    # --- MERGE CƠ BẢN ---
    df_content = pd.merge(df_games, df_tags[["appid", "all_tags", "steamspy_tags"]], on="appid", how="left")
    try:
        df_content = pd.merge(df_content, df_reqs[["appid", "minimum"]], on="appid", how="left")
    except Exception:
        df_content["minimum"] = "8 GB RAM"

    # --- FIX HEADER IMAGE (trước khi merge thêm, để appid còn tồn tại) ---
    def get_img(row):
        if "header_image" in row and pd.notna(row["header_image"]):
            return row["header_image"]
        return f"https://cdn.akamai.steamstatic.com/steam/apps/{row['appid']}/header.jpg"

    df_content["header_image"] = df_content.apply(get_img, axis=1)

    # --- MERGE DỮ LIỆU BỔ SUNG ---
    # 1. Media (video/trailer)
    if df_media is not None:
        print("   -> Merging media (video/trailer)...")
        df_content["appid"] = pd.to_numeric(df_content["appid"], errors="coerce").fillna(0).astype(int)
        if "steam_appid" in df_media.columns:
            df_media["steam_appid"] = pd.to_numeric(df_media["steam_appid"], errors="coerce").fillna(0).astype(int)
            if "movies" in df_media.columns:
                df_content = pd.merge(
                    df_content, df_media[["steam_appid", "movies"]], left_on="appid", right_on="steam_appid", how="left"
                )
                if "steam_appid" in df_content.columns:
                    df_content.drop("steam_appid", axis=1, inplace=True)

    # 2. Description (mô tả chi tiết)
    if df_desc is not None:
        print("   -> Merging description...")
        id_col = "appid" if "appid" in df_desc.columns else "steam_appid"
        df_desc[id_col] = pd.to_numeric(df_desc[id_col], errors="coerce").fillna(0).astype(int)

        desc_col = "short_description" if "short_description" in df_desc.columns else "description"
        if desc_col in df_desc.columns:
            df_content = pd.merge(df_content, df_desc[[id_col, desc_col]], left_on="appid", right_on=id_col, how="left")
            if desc_col != "short_description":
                df_content.rename(columns={desc_col: "short_description"}, inplace=True)
            if id_col != "appid" and id_col in df_content.columns:
                df_content.drop(id_col, axis=1, inplace=True)

    # --- FILL NA ---
    for col in ["all_tags", "genres", "developer"]:
        if col not in df_content.columns:
            df_content[col] = "Unknown"
        df_content[col] = df_content[col].fillna("Unknown")

    # --- SENTIMENT_SCORE từ ratings thật ---
    if "positive_ratings" in df_content.columns and "negative_ratings" in df_content.columns:
        pos = pd.to_numeric(df_content["positive_ratings"], errors="coerce").fillna(0)
        neg = pd.to_numeric(df_content["negative_ratings"], errors="coerce").fillna(0)
        df_content["sentiment_score"] = (pos / (pos + neg + 1)).astype(float)
        df_content["total_reviews"] = (pos + neg).astype(float)
        # ưu tiên game nhiều review để collab signal ổn hơn
        df_content = df_content.sort_values("total_reviews", ascending=False)
    else:
        df_content["sentiment_score"] = 0.8

    # --- SOUP (metadata gộp) ---
    df_content["soup"] = (
        df_content["developer"].apply(clean_text) + " " + df_content["all_tags"] + " " + df_content["genres"]
    )

    # --- LIMIT BY RAM ---
    limit = pick_limit_by_ram()
    df_content_limited = df_content.head(limit).reset_index(drop=True).copy()
    print(
        f"   RAM available ~{psutil.virtual_memory().available/(1024**3):.1f}GB -> limit games = {len(df_content_limited)}"
    )
    return df_content_limited, df_users


def build_user_item_sparse(df_users: pd.DataFrame, df_content: pd.DataFrame, cfg: Config):
    """
    Tránh pivot_table (dense). Build sparse matrix từ triplets (user_idx, item_idx, rating).
    """
    df = df_users[df_users["Action"] == "play"].copy()
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce").fillna(0.0)

    valid_games = set(df_content["name"].astype(str).tolist())
    df = df[df["Game"].isin(valid_games)].copy()

    # lọc game đủ người chơi để giảm noise
    game_counts = df.groupby("Game")["UserID"].count().sort_values(ascending=False)
    popular_games = set(game_counts[game_counts >= cfg.min_users_per_game].index.tolist())
    df = df[df["Game"].isin(popular_games)].copy()

    # top games list
    top_games_list = game_counts.head(100).index.tolist()

    # map item order theo df_content
    name_to_item = {str(n): int(i) for i, n in enumerate(df_content["name"].astype(str).tolist())}
    df["ItemIdx"] = df["Game"].map(name_to_item).astype("int32")

    # user indices (dense codes)
    df["UserIdx"] = df["UserID"].astype("int64").astype("category").cat.codes.astype("int32")

    # rating
    df["Rating"] = df["Hours"].apply(convert_hours_to_rating).astype("float32")

    # aggregate duplicates user-item: keep max rating
    df = df.groupby(["UserIdx", "ItemIdx"], as_index=False)["Rating"].max()

    n_users = int(df["UserIdx"].max()) + 1 if len(df) else 0
    n_items = len(df_content)

    mat = sparse.coo_matrix(
        (df["Rating"].to_numpy(), (df["UserIdx"].to_numpy(), df["ItemIdx"].to_numpy())),
        shape=(n_users, n_items),
        dtype=np.float32,
    ).tocsr()

    return mat, top_games_list


def compute_hours_boost(df_users: pd.DataFrame, df_content: pd.DataFrame, cfg: Config) -> np.ndarray:
    df = df_users[df_users["Action"] == "play"].copy()
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce").fillna(0.0)
    valid_games = set(df_content["name"].astype(str).tolist())
    df = df[df["Game"].isin(valid_games)].copy()
    game_counts = df.groupby("Game")["UserID"].count()
    popular_games = set(game_counts[game_counts >= cfg.min_users_per_game].index.tolist())
    df = df[df["Game"].isin(popular_games)].copy()

    hours_mean = df.groupby("Game")["Hours"].mean()
    # align to df_content order
    hours_vec = df_content["name"].astype(str).map(hours_mean).fillna(10.0).astype(float).to_numpy()

    # boost: log1p relative median
    med = float(np.median(hours_vec)) if len(hours_vec) else 1.0
    med = med if med > 0 else 1.0
    hours_boost = np.log1p(hours_vec / med)
    # normalize 0..1
    mn, mx = float(hours_boost.min()), float(hours_boost.max())
    if mx > mn:
        hours_boost = (hours_boost - mn) / (mx - mn)
    else:
        hours_boost = np.zeros_like(hours_boost)
    return hours_boost.astype(np.float16)


def compute_hybrid_matrix(df_content: pd.DataFrame, user_item: sparse.csr_matrix, hours_boost: np.ndarray, cfg: Config):
    n_games = len(df_content)

    print("[2/4] Content similarity (TF-IDF)...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=cfg.max_features)
    tfidf_matrix = tfidf.fit_transform(df_content["soup"].astype(str))
    content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix).astype(np.float16)

    print("[3/4] Collaborative similarity (item-item cosine on sparse user matrix)...")
    # item-user matrix
    item_user = user_item.T.tocsr()
    collab_sim = cosine_similarity(item_user, item_user).astype(np.float16)

    print("[3/4] Hours boost similarity...")
    hours_matrix = hours_boost.reshape(-1, 1) @ hours_boost.reshape(1, -1)
    mx = float(hours_matrix.max()) if hours_matrix.size else 1.0
    mx = mx if mx != 0 else 1.0
    hours_sim = np.clip(hours_matrix / mx, 0, 1).astype(np.float16)

    print("[4/4] Sentiment similarity (from real ratings if available)...")
    sent = df_content.get("sentiment_score", pd.Series([0.8] * n_games)).fillna(0.8).astype(float).to_numpy()
    sent = np.clip(sent, 0.0, 1.0).astype(np.float16)
    sentiment_sim = np.clip(sent.reshape(-1, 1) @ sent.reshape(1, -1), 0, 1).astype(np.float16)

    hybrid = (
        cfg.w_collab * collab_sim + cfg.w_hours * hours_sim + cfg.w_content * content_sim + cfg.w_sentiment * sentiment_sim
    ).astype(np.float16)

    # clip to [0,1] for stability
    hybrid = np.clip(hybrid, 0, 1).astype(np.float16)
    return hybrid, tfidf


def save_production_files(df_content: pd.DataFrame, hybrid: np.ndarray, top_games: list, tfidf, cfg: Config) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. CORE: Hybrid Similarity (Ma trận NxN)
    with open(f"{cfg.output_dir}/hybrid_similarity.pkl", "wb") as f:
        pickle.dump(hybrid, f)

    # 2. METADATA (BAO GỒM ĐẦY ĐỦ CÁC CỘT CHO APP.PY)
    cols_to_keep = [
        "appid",
        "name",
        "genres",
        "price",
        "developer",
        "minimum",
        "header_image",
        "movies",  # video/trailer
        "short_description",  # mô tả game
        "steamspy_tags",  # tags dict
        "sentiment_score",
        "positive_ratings",
        "negative_ratings",
        "release_date",
    ]
    cols_final = [c for c in cols_to_keep if c in df_content.columns]
    df_content[cols_final].to_pickle(f"{cfg.output_dir}/games_metadata.pkl")
    print(f"   OK - Saved games_metadata.pkl ({len(cols_final)} columns: {cols_final})")

    # 3. UTILS
    with open(f"{cfg.output_dir}/top_games_list.pkl", "wb") as f:
        pickle.dump(top_games, f)

    with open(f"{cfg.output_dir}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    # SIZE CHECK
    size_mb = os.path.getsize(f"{cfg.output_dir}/hybrid_similarity.pkl") / (1024**2)
    print(f"   OK - PRODUCTION READY! Hybrid Matrix Size: {size_mb:.1f}MB")
    print(f"   Files saved to: {os.path.abspath(cfg.output_dir)}")


def main() -> None:
    if _is_colab():
        # nếu là Colab, tự skip args kiểu -f kernel.json
        args_list = [a for a in sys.argv[1:] if not a.startswith("-f")]
        sys.argv = [sys.argv[0]] + args_list

    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="Thư mục chứa data (steam.csv, ...)")
    p.add_argument("--output_dir", type=str, default="models", help="Thư mục lưu model")
    p.add_argument("--mount_drive", action="store_true", help="(Colab) Tự mount Google Drive")
    args, _ = p.parse_known_args()

    _maybe_mount_drive(args.mount_drive)
    data_dir = _resolve_data_dir(args.data_dir)

    print("=" * 60)
    print("TRAINING (PRECOMPUTE FULL MATRIX)")
    print("=" * 60)
    print(f"data_dir={data_dir} | output_dir={args.output_dir}")
    print("=" * 60)

    cfg = Config(data_dir=data_dir, output_dir=args.output_dir)

    df_content, df_users = load_and_process_data(cfg)
    user_item_sparse, top_games_list = build_user_item_sparse(df_users, df_content, cfg)
    hours_boost = compute_hours_boost(df_users, df_content, cfg)
    hybrid_matrix, tfidf = compute_hybrid_matrix(df_content, user_item_sparse, hours_boost, cfg)
    save_production_files(df_content, hybrid_matrix, top_games_list, tfidf, cfg)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
