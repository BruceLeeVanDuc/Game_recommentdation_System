"""
ĐÁNH GIÁ HỆ THỐNG GỢI Ý - 4 METRICS
RMSE, MAE, Precision@K, Recall@K

Lưu ý quan trọng về format model:
- Nếu `models/hybrid_similarity.pkl` là MA TRẬN similarity (NxN): RMSE/MAE đo sai số
  giữa similarity dự đoán và "ground truth" Jaccard(genres).
- Nếu `models/hybrid_similarity.pkl` là DICT top-N (chỉ chứa indices, không có score):
  không có predicted score để tính RMSE/MAE theo nghĩa truyền thống.
  Script sẽ chuyển sang RMSE/MAE nhị phân: y_pred=1 nếu item được recommend, 0 nếu không;
  y_true=1 nếu relevant (có ít nhất 1 genre chung), 0 nếu không.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections import defaultdict
from math import sqrt
from typing import DefaultDict, Dict, List, Literal, Sequence, Set, Tuple, Union, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


Similarity = Union[Dict[int, np.ndarray], np.ndarray]
RelevanceMode = Literal["overlap", "jaccard"]
EvalMode = Literal["genre", "user_holdout"]


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
        # Not on colab or cannot mount; ignore
        return


def _resolve_model_paths(sim_path: str, meta_path: str) -> Tuple[str, str]:
    """
    Hỗ trợ Colab: nếu user không truyền path đúng, tự dò trong các path phổ biến.
    """
    sim_p = Path(sim_path)
    meta_p = Path(meta_path)

    if sim_p.exists() and meta_p.exists():
        return str(sim_p), str(meta_p)

    candidates = [
        Path("models"),
        Path("/content/models"),
        Path("/content/drive/MyDrive/models"),
        Path("/content/drive/MyDrive/models_output"),
        Path("/content/drive/MyDrive/KHDL_CVS/models_output"),
        Path("/content/drive/MyDrive/KHDL_CVS/models"),
    ]

    def find_file(file_name: str) -> Path | None:
        # check direct candidates
        for base in candidates:
            p = base / file_name
            if p.exists():
                return p
        # check recursively in Drive (bounded)
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            # first few matches only, to avoid heavy scan
            try:
                for p in drive_root.rglob(file_name):
                    return p
            except Exception:
                return None
        return None

    if not sim_p.exists():
        found = find_file(sim_p.name)
        if found is not None:
            sim_p = found

    if not meta_p.exists():
        found = find_file(meta_p.name)
        if found is not None:
            meta_p = found

    return str(sim_p), str(meta_p)


def _resolve_user_data_path(user_data_path: str) -> str:
    """
    Hỗ trợ Colab: tự dò steam-200k.csv nếu path mặc định không tồn tại.
    """
    p = Path(user_data_path)
    if p.exists():
        return str(p)

    candidates = [
        Path("."),
        Path("data"),
        Path("/content"),
        Path("/content/data"),
        Path("/content/drive/MyDrive"),
        Path("/content/drive/MyDrive/KHDL_CVS"),
    ]

    # 1) thử match theo basename trong các folder phổ biến
    for base in candidates:
        cand = base / p.name
        if cand.exists():
            return str(cand)

    # 2) thử match "data/<basename>"
    for base in candidates:
        cand = base / "data" / p.name
        if cand.exists():
            return str(cand)

    # 3) scan trong Drive (bounded: lấy match đầu tiên)
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        try:
            for found in drive_root.rglob(p.name):
                return str(found)
        except Exception:
            pass

    return str(p)  # fallback (sẽ fail rõ ràng ở chỗ đọc file)

def load_similarity(path: str) -> Similarity:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_metadata(path: str) -> pd.DataFrame:
    return pd.read_pickle(path).reset_index(drop=True)


def load_user_data_steam200k(path: str) -> pd.DataFrame:
    """
    steam-200k.csv format: UserID, Game, Action, Hours, Unused
    (không có header)
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["UserID", "Game", "Action", "Hours", "Unused"],
    )
    return df


def add_dynamic_rating(df_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo cột Rating 1..5 dựa trên phân vị Hours theo từng game (giống train_model_safe.py).
    Vectorized để chạy nhanh.
    """
    df = df_plays.copy()
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce").fillna(0.0)

    q = (
        df.groupby("Game")["Hours"]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: "q25", 0.5: "q50", 0.75: "q75"})
    )
    df = df.join(q, on="Game")

    # fallback: game thiếu stats => rating=3
    q25 = df["q25"]
    q50 = df["q50"]
    q75 = df["q75"]
    h = df["Hours"]

    conds = [
        h < q25,
        h < q50,
        h < q75,
        h < (q75 * 1.5),
    ]
    choices = [1.0, 2.0, 3.0, 4.0]
    df["Rating"] = np.select(conds, choices, default=5.0).astype(float)
    df.loc[q25.isna() | q50.isna() | q75.isna(), "Rating"] = 3.0

    return df.drop(columns=["q25", "q50", "q75"])


def build_name_to_index(df_meta: pd.DataFrame, name_col: str = "name") -> Dict[str, int]:
    if name_col not in df_meta.columns:
        return {}
    # giữ nguyên string như trong metadata (train cũng match exact)
    return {str(name): int(i) for i, name in enumerate(df_meta[name_col].tolist())}


def evaluate_user_holdout(
    sim_data: Similarity,
    df_meta: pd.DataFrame,
    user_data_path: str,
    k: int,
    test_ratio: float,
    min_interactions: int,
    max_users: int,
    seed: int,
) -> Dict[str, float]:
    """
    Đánh giá "model đã train" theo user holdout:
    - Split interactions per-user thành train/test
    - Recommend top-K từ các item train của user
    - Precision@K / Recall@K đo trên test items
    - RMSE/MAE: dự đoán rating test item bằng neighborhood item-item (weighted average)
      (nếu sim_data là dict topK thì weight theo rank 1/(rank+1))
    """
    print("\n--- USER HOLDOUT EVAL ---")
    print(f"user_data_path: {user_data_path}")
    print(f"test_ratio: {test_ratio} | min_interactions: {min_interactions} | max_users: {max_users}")

    df_users = load_user_data_steam200k(user_data_path)
    df_users = df_users[df_users["Action"] == "play"].copy()

    # chỉ giữ game có trong metadata/model
    name_to_idx = build_name_to_index(df_meta, "name")
    df_users = df_users[df_users["Game"].isin(name_to_idx)].copy()
    if df_users.empty:
        raise ValueError("No user interactions match metadata game names. Check dataset/path.")

    df_users = add_dynamic_rating(df_users)
    df_users["ItemIdx"] = df_users["Game"].map(name_to_idx).astype(int)

    # nếu có trùng user-item, giữ rating cao nhất
    df_users = (
        df_users.groupby(["UserID", "ItemIdx"], as_index=False)["Rating"]
        .max()
        .sort_values(["UserID"])
    )

    # gom theo user
    user_groups = df_users.groupby("UserID")["ItemIdx"].apply(list)
    # build dict user -> {item: rating}
    user_ratings = df_users.groupby("UserID")[["ItemIdx", "Rating"]].apply(
        lambda g: dict(zip(g["ItemIdx"], g["Rating"]))
    ).to_dict()

    # lọc user đủ tương tác
    users = [u for u, items in user_groups.items() if len(items) >= min_interactions]
    if not users:
        raise ValueError("No users with enough interactions for holdout split.")

    rng = np.random.default_rng(seed)
    if len(users) > max_users:
        users = rng.choice(np.array(users, dtype=object), size=max_users, replace=False).tolist()

    precisions: List[float] = []
    recalls: List[float] = []
    y_true: List[float] = []
    y_pred: List[float] = []

    # pre-calc global mean rating for fallback
    global_mean = float(df_users["Rating"].mean())

    for u in users:
        items = list(user_groups[u])
        rating_map: Dict[int, float] = user_ratings.get(u, {})
        if len(items) < 2:
            continue

        rng.shuffle(items)
        test_size = max(1, int(round(len(items) * test_ratio)))
        test_items = set(items[:test_size])
        train_items = items[test_size:]
        if not train_items:
            # đảm bảo còn train
            train_items = items[:-1]
            test_items = {items[-1]}

        train_set = set(train_items)
        train_ratings = [float(rating_map.get(i, 3.0)) for i in train_items]
        user_mean = float(np.mean(train_ratings)) if train_ratings else global_mean

        # build candidate scores for recommendation
        cand_scores: DefaultDict[int, float] = defaultdict(float)
        # also build rec rank maps for rating prediction
        train_rec_rank: Dict[int, Dict[int, int]] = {}

        for ti in train_items:
            ti = int(ti)
            recs = get_recs(sim_data, ti, k=100)
            # rank map for fast lookup
            rank_map: Dict[int, int] = {}
            for rank, r in enumerate(recs):
                r = int(r)
                rank_map[r] = rank + 1  # 1-based
                if r in train_set:
                    continue
                # weight: similarity score if matrix, else rank-based
                if is_topk_dict(sim_data):
                    w = 1.0 / (rank + 1)
                else:
                    w = float(np.asarray(sim_data[ti][r]).item())
                cand_scores[r] += w * float(rating_map.get(ti, 3.0))
            train_rec_rank[ti] = rank_map

        # top-K recommend
        if cand_scores:
            top = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
            recs_u = [i for i, _s in top if i not in train_set][:k]
        else:
            recs_u = []

        hit = sum(1 for i in recs_u if i in test_items)
        precisions.append(hit / k if k > 0 else 0.0)
        recalls.append(hit / len(test_items) if test_items else 0.0)

        # rating prediction for each test item (neighborhood weighted avg)
        for ti_test in test_items:
            ti_test = int(ti_test)
            num = 0.0
            den = 0.0
            for ti in train_items:
                ti = int(ti)
                rank_map = train_rec_rank.get(ti, {})
                if ti_test not in rank_map:
                    continue
                rank_pos = rank_map[ti_test]
                if is_topk_dict(sim_data):
                    w = 1.0 / rank_pos
                else:
                    w = float(np.asarray(sim_data[ti][ti_test]).item())
                num += w * float(rating_map.get(ti, 3.0))
                den += abs(w)

            pred = (num / den) if den > 0 else user_mean
            pred = float(np.clip(pred, 1.0, 5.0))
            true = float(rating_map.get(ti_test, 3.0))

            y_pred.append(pred)
            y_true.append(true)

    precision = float(np.mean(precisions)) if precisions else 0.0
    recall = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    rmse = sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0.0
    mae = mean_absolute_error(y_true, y_pred) if y_true else 0.0

    return {
        "Eval_Mode": 1.0,  # placeholder numeric (csv-friendly); see printed mode text
        "Users_Evaluated": float(len(users)),
        "Test_Interactions": float(len(y_true)),
        "RMSE": float(rmse),
        "MAE": float(mae),
        f"Precision@{k}": precision,
        f"Recall@{k}": recall,
        "F1_Score": float(f1),
    }

def _split_genres(value: object) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and np.isnan(value):
        return set()
    raw = str(value).strip()
    if not raw:
        return set()
    parts = [p.strip() for p in raw.split(";")]
    return {p for p in parts if p}


def build_genre_sets(df: pd.DataFrame, col: str = "genres") -> List[Set[str]]:
    if col not in df.columns:
        return [set() for _ in range(len(df))]
    return [_split_genres(v) for v in df[col].tolist()]


def build_inverted_index(genre_sets: Sequence[Set[str]]) -> DefaultDict[str, List[int]]:
    inv: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, gs in enumerate(genre_sets):
        for g in gs:
            inv[g].append(idx)
    return inv


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def relevant_set_for_query(
    query_idx: int,
    genre_sets: Sequence[Set[str]],
    inv: DefaultDict[str, List[int]],
    mode: RelevanceMode = "overlap",
    jaccard_min: float = 0.0,
) -> Set[int]:
    gs = genre_sets[query_idx]
    if not gs:
        return set()

    # Candidate pool: all items sharing at least 1 genre (fast via inverted index)
    cand: Set[int] = set()
    for g in gs:
        cand.update(inv.get(g, []))
    cand.discard(query_idx)

    if mode == "overlap":
        return cand

    # mode == "jaccard": filter by Jaccard threshold
    if jaccard_min <= 0:
        return cand

    out: Set[int] = set()
    for j in cand:
        if jaccard(gs, genre_sets[j]) >= jaccard_min:
            out.add(j)
    return out


def is_topk_dict(sim_data: object) -> bool:
    return isinstance(sim_data, dict)


def get_recs(sim_data: Similarity, idx: int, k: int) -> List[int]:
    if is_topk_dict(sim_data):
        arr = sim_data.get(idx, np.array([], dtype=np.int32))
        recs = arr[:k].tolist() if hasattr(arr, "tolist") else list(arr[:k])  # type: ignore[arg-type]
        return [int(r) for r in recs if int(r) != idx]

    row = np.asarray(sim_data[idx]).ravel()
    n = row.shape[0]
    kk = min(k + 1, n)
    top = np.argpartition(-row, kk - 1)[:kk]
    top = top[np.argsort(-row[top])]
    out: List[int] = []
    for j in top:
        j = int(j)
        if j == idx:
            continue
        out.append(j)
        if len(out) >= k:
            break
    return out


def sample_query_indices(rng: np.random.Generator, genre_sets: Sequence[Set[str]], n: int) -> np.ndarray:
    valid = np.array([i for i, gs in enumerate(genre_sets) if gs], dtype=np.int32)
    if len(valid) == 0:
        return np.array([], dtype=np.int32)
    return rng.choice(valid, size=min(n, len(valid)), replace=False)


def precision_recall_at_k(
    sim_data: Similarity,
    genre_sets: Sequence[Set[str]],
    inv: DefaultDict[str, List[int]],
    k: int,
    n_queries: int,
    seed: int,
    relevance_mode: RelevanceMode = "overlap",
    jaccard_min: float = 0.0,
    use_k_denominator: bool = True,
) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    queries = sample_query_indices(rng, genre_sets, n_queries)
    if len(queries) == 0:
        return 0.0, 0.0, 0

    precisions: List[float] = []
    recalls: List[float] = []
    used = 0

    for q in queries:
        q = int(q)
        rel = relevant_set_for_query(
            q,
            genre_sets,
            inv,
            mode=relevance_mode,
            jaccard_min=jaccard_min,
        )
        if not rel:
            continue

        recs = get_recs(sim_data, q, k)
        hit = sum(1 for r in recs if int(r) in rel)

        denom_p = k if use_k_denominator else len(recs)
        p = hit / denom_p if denom_p > 0 else 0.0
        r = hit / len(rel) if len(rel) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        used += 1

    if used == 0:
        return 0.0, 0.0, 0

    return float(np.mean(precisions)), float(np.mean(recalls)), used


def rmse_mae(
    sim_data: Similarity,
    genre_sets: Sequence[Set[str]],
    inv: DefaultDict[str, List[int]],
    n_queries: int,
    n_random_per_query: int,
    seed: int,
    relevance_mode: RelevanceMode = "overlap",
    jaccard_min: float = 0.0,
    dict_neg_per_pos: int = 1,
) -> Tuple[float, float, int, str]:
    rng = np.random.default_rng(seed)
    queries = sample_query_indices(rng, genre_sets, n_queries)
    if len(queries) == 0:
        return 0.0, 0.0, 0, "no_valid_queries"

    y_pred: List[float] = []
    y_true: List[float] = []
    n_items = len(genre_sets)

    if not is_topk_dict(sim_data):
        # matrix mode: y_pred=sim, y_true=jaccard(genres)
        for q in queries:
            q = int(q)
            candidates = rng.choice(np.arange(n_items, dtype=np.int32), size=min(n_random_per_query, n_items), replace=False)
            for j in candidates:
                j = int(j)
                if j == q:
                    continue
                y_true.append(jaccard(genre_sets[q], genre_sets[j]))
                y_pred.append(float(np.asarray(sim_data[q][j]).item()))

        if not y_pred:
            return 0.0, 0.0, 0, "matrix_no_pairs"

        p = np.asarray(y_pred, dtype=np.float64)
        p_min, p_max = float(np.min(p)), float(np.max(p))
        if p_max > p_min:
            p = (p - p_min) / (p_max - p_min)
        p = np.clip(p, 0.0, 1.0)
        y_pred = p.tolist()
        mode = "matrix_jaccard"
    else:
        # dict mode: binary proxy
        for q in queries:
            q = int(q)
            recs = get_recs(sim_data, q, k=50)
            if not recs:
                continue
            rec_set = set(int(r) for r in recs if 0 <= int(r) < n_items and int(r) != q)
            if not rec_set:
                continue

            pos = list(rec_set)
            rng.shuffle(pos)
            pos = pos[: min(len(pos), n_random_per_query)]

            need_neg = len(pos) * max(1, dict_neg_per_pos)
            pool = np.array([i for i in range(n_items) if i != q and i not in rec_set], dtype=np.int32)
            neg = rng.choice(pool, size=min(need_neg, len(pool)), replace=False).astype(int).tolist() if len(pool) else []

            rel = relevant_set_for_query(
                q,
                genre_sets,
                inv,
                mode=relevance_mode,
                jaccard_min=jaccard_min,
            )
            for j in pos:
                y_pred.append(1.0)
                y_true.append(1.0 if j in rel else 0.0)
            for j in neg:
                y_pred.append(0.0)
                y_true.append(1.0 if j in rel else 0.0)

        if not y_pred:
            return 0.0, 0.0, 0, "dict_no_pairs"
        mode = "dict_binary"

    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return float(rmse), float(mae), len(y_true), mode


def evaluate(
    k: int = 10,
    n: int = 200,
    seed: int = 42,
    sim_path: str = "models/hybrid_similarity.pkl",
    meta_path: str = "models/games_metadata.pkl",
    mount_drive: bool = False,
    eval_mode: EvalMode = "genre",
    user_data_path: str = "data/steam-200k.csv",
    test_ratio: float = 0.2,
    min_interactions: int = 5,
    max_users: int = 2000,
    relevance_mode: RelevanceMode = "overlap",
    jaccard_min: float = 0.0,
) -> dict:
    if _is_colab():
        _maybe_mount_drive(mount_drive)

    sim_path, meta_path = _resolve_model_paths(sim_path, meta_path)
    user_data_path = _resolve_user_data_path(user_data_path)

    print("Loading model & metadata...")
    print(f"sim_path:  {sim_path}")
    print(f"meta_path: {meta_path}")
    sim_data = load_similarity(sim_path)
    df = load_metadata(meta_path)

    # USER HOLDOUT: đánh giá đúng kiểu "model đã train" dựa trên user-play
    if eval_mode == "user_holdout":
        results = evaluate_user_holdout(
            sim_data=sim_data,
            df_meta=df,
            user_data_path=user_data_path,
            k=k,
            test_ratio=test_ratio,
            min_interactions=min_interactions,
            max_users=max_users,
            seed=seed,
        )
        print(f"\n{'='*60}")
        print("RESULTS (USER HOLDOUT)")
        print(f"{'='*60}\n")
        print(f"Users evaluated: {int(results.get('Users_Evaluated', 0))}")
        print(f"Test interactions: {int(results.get('Test_Interactions', 0))}\n")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAE : {results['MAE']:.4f}\n")
        print(f"Precision@{k}: {results[f'Precision@{k}']:.4f}")
        print(f"Recall@{k}:    {results[f'Recall@{k}']:.4f}")
        print(f"F1-Score:      {results['F1_Score']:.4f}\n")

        out_path = "models/evaluation_results.csv"
        pd.DataFrame([results]).to_csv(out_path, index=False)
        print(f"Saved to {out_path}\n")
        return results

    # GENRE MODE: đánh giá item-item theo genres (debug nhanh)
    genre_sets = build_genre_sets(df, "genres")
    inv = build_inverted_index(genre_sets)

    fmt = "dict(topK)" if is_topk_dict(sim_data) else "matrix"
    print(f"Loaded {len(df)} games | similarity format: {fmt}")

    print(f"\n{'='*60}")
    print("EVALUATION (RMSE, MAE, Precision@K, Recall@K)")
    print(f"{'='*60}")
    print(f"Queries: {n} | Top-K: {k} | Seed: {seed}")
    if relevance_mode == "overlap":
        print("Relevance: share >= 1 genre (very broad -> Recall@K often tiny)")
    else:
        print(f"Relevance: Jaccard(genres) >= {jaccard_min}")
    print(f"{'='*60}\n")

    rmse_v, mae_v, n_pairs, rmse_mode = rmse_mae(
        sim_data=sim_data,
        genre_sets=genre_sets,
        inv=inv,
        n_queries=n,
        n_random_per_query=10,
        seed=seed,
        relevance_mode=relevance_mode,
        jaccard_min=jaccard_min,
        dict_neg_per_pos=1,
    )

    precision, recall, used_queries = precision_recall_at_k(
        sim_data=sim_data,
        genre_sets=genre_sets,
        inv=inv,
        k=k,
        n_queries=n,
        seed=seed,
        relevance_mode=relevance_mode,
        jaccard_min=jaccard_min,
        use_k_denominator=True,
    )

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    print(f"RMSE: {rmse_v:.4f} | mode={rmse_mode} | pairs={n_pairs}")
    print(f"MAE : {mae_v:.4f} | mode={rmse_mode} | pairs={n_pairs}\n")
    print(f"Precision@{k}: {precision:.4f} ({precision*100:.2f}%) | queries_used={used_queries}")
    print(f"Recall@{k}:    {recall:.4f} ({recall*100:.2f}%) | queries_used={used_queries}")
    print(f"F1-Score:      {f1:.4f} ({f1*100:.2f}%)\n")

    results = {
        "RMSE": rmse_v,
        "MAE": mae_v,
        f"Precision@{k}": float(precision),
        f"Recall@{k}": float(recall),
        "F1_Score": float(f1),
        "Queries_Used": float(used_queries),
        "Pairs_For_RMSE": float(n_pairs),
    }

    out_path = "models/evaluation_results.csv"
    pd.DataFrame([results]).to_csv(out_path, index=False)
    print(f"Saved to {out_path}\n")
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=10, help="Top-K")
    p.add_argument("--n", type=int, default=200, help="Số query samples (chỉ lấy các game có genres)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--sim_path", type=str, default="models/hybrid_similarity.pkl", help="Path tới similarity pickle")
    p.add_argument("--meta_path", type=str, default="models/games_metadata.pkl", help="Path tới games metadata pickle")
    p.add_argument("--mount_drive", action="store_true", help="(Colab) Tự mount Google Drive vào /content/drive")
    p.add_argument(
        "--eval_mode",
        type=str,
        default="user_holdout",
        choices=["user_holdout", "genre"],
        help="Cách đánh giá: user_holdout (đúng kiểu model đã train) hoặc genre (debug nhanh theo genres)",
    )
    p.add_argument("--user_data_path", type=str, default="data/steam-200k.csv", help="Path tới steam-200k.csv")
    p.add_argument("--test_ratio", type=float, default=0.2, help="Tỷ lệ test per-user (0-1)")
    p.add_argument("--min_interactions", type=int, default=5, help="User phải có ít nhất N interactions để đánh giá")
    p.add_argument("--max_users", type=int, default=2000, help="Giới hạn số user để chạy nhanh (sample)")
    p.add_argument(
        "--relevance_mode",
        type=str,
        default="overlap",
        choices=["overlap", "jaccard"],
        help="Cách định nghĩa 'relevant' cho Precision/Recall: overlap=share >=1 genre; jaccard=Jaccard(genres) >= --jaccard_min",
    )
    p.add_argument(
        "--jaccard_min",
        type=float,
        default=0.3,
        help="Ngưỡng Jaccard tối thiểu khi --relevance_mode=jaccard (gợi ý: 0.2~0.4)",
    )
    # Colab/Jupyter thường inject thêm args như: -f /path/to/kernel.json
    # Nếu dùng `%run evaluate_final.py ...` sẽ bị crash nếu parse_args() strict.
    args, _unknown = p.parse_known_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        k=args.k,
        n=args.n,
        seed=args.seed,
        sim_path=args.sim_path,
        meta_path=args.meta_path,
        mount_drive=args.mount_drive,
        eval_mode=args.eval_mode,
        user_data_path=args.user_data_path,
        test_ratio=args.test_ratio,
        min_interactions=args.min_interactions,
        max_users=args.max_users,
        relevance_mode=args.relevance_mode,
        jaccard_min=args.jaccard_min,
    )


