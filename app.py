import streamlit as st
import pandas as pd
import pickle
import os
import ast
from deep_translator import GoogleTranslator
from datetime import datetime

# 1. CẤU HÌNH TRANG WEB
st.set_page_config(
    page_title="Steam Game Store",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho đẹp (Dark theme giống Steam)
st.markdown("""
<style>
    .stApp { background-color: #1b2838; color: #c7d5e0; }
    div[data-testid="stMetricValue"] { color: #66c0f4; }
    .game-card { 
        background-color: #16202d; 
        padding: 15px; 
        border-radius: 5px; 
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .game-card:hover {
        transform: scale(1.02);
        background-color: #1e2d3f;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .section-title {
        color: #66c0f4 !important;
        font-size: 24px !important;
        font-weight: bold !important;
        margin-top: 30px !important;
        margin-bottom: 15px !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #66c0f4 !important;
        font-size: 18px !important;
        font-weight: bold !important;
    }
    .stButton>button {
        background-color: #5c7e10;
        color: white;
        border-radius: 3px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #7ba82d;
    }
    /* Màu đen cho tiêu đề Bộ lọc và Thống kê trong sidebar */
    div[data-testid="stSidebar"] h2 {
        color: #000000 !important;
    }
    div[data-testid="stSidebar"] h3 {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. LOAD DATA
@st.cache_resource
def load_translator():
    """Khởi tạo translator (cache để tái sử dụng)"""
    return GoogleTranslator(source='en', target='vi')

@st.cache_resource
def load_data():
    base_dir = "models"
    try:
        # Dùng lightweight version để deploy (7.6MB thay vì 122MB)
        with open(f'{base_dir}/hybrid_similarity_lightweight.pkl', 'rb') as f:
            sim_matrix = pickle.load(f)
        
        df = pd.read_pickle(f'{base_dir}/games_metadata_lightweight.pkl')
        df = df.reset_index(drop=True)

        # Kiểm tra các cột quan trọng
        if 'header_image' not in df.columns:
            st.warning("⚠️ Dữ liệu chưa có ảnh. Chạy: python add_images_to_model.py")
            df['header_image'] = "https://via.placeholder.com/460x215?text=No+Image"
        
        if 'price' not in df.columns: df['price'] = 0.0
        if 'developer' not in df.columns: df['developer'] = "Unknown"
        if 'genres' not in df.columns: df['genres'] = "Game"
        
        # Xử lý release_date để sắp xếp
        if 'release_date' in df.columns:
            df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        return sim_matrix, df
    except FileNotFoundError:
        st.error("❌ Lỗi: Không tìm thấy file trong thư mục 'models'. Hãy chạy train trước!")
        return None, None

try:
    sim_matrix, df_games = load_data()
    translator = load_translator()
except Exception as e:
    st.error(f"Lỗi khi load data: {e}")
    sim_matrix, df_games = None, None
    translator = None

# 3. HÀM HỖ TRỢ
def display_game_card(game_data, show_description=False, show_view_button=True):
    """Hiển thị card game với nút xem ở góc phải"""
    # Hiển thị ảnh
    st.image(game_data['header_image'], use_container_width=True)
    
    # Tên game
    game_name_display = game_data['name'][:35] + '...' if len(game_data['name']) > 35 else game_data['name']
    st.markdown(f"**{game_name_display}**")
    
    # Thể loại
    genre_first = game_data['genres'].split(';')[0] if ';' in str(game_data['genres']) else game_data['genres']
    st.caption(f"Thể loại: {genre_first}")
    
    # Đánh giá
    if 'positive_ratings' in game_data and 'negative_ratings' in game_data:
        total = game_data['positive_ratings'] + game_data['negative_ratings']
        if total > 0:
            positive_pct = (game_data['positive_ratings'] / total) * 100
            if positive_pct >= 80:
                st.caption(f"Đánh giá: {positive_pct:.0f}% tích cực")
            else:
                st.caption(f"Đánh giá:{positive_pct:.0f}% tích cực")
    
    # Giá
    if game_data['price'] == 0:
        st.markdown("**🆓 Miễn phí**")
    else:
        st.markdown(f"**💰 ${game_data['price']}**")
    
    # Nút xem chi tiết - đặt ở dưới cùng
    if show_view_button:
        # Tạo key duy nhất cho mỗi game
        import hashlib
        game_key = hashlib.md5(game_data['name'].encode()).hexdigest()[:8]
        
        if st.button("🔍 Xem chi tiết", key=f"view_{game_key}", 
                   use_container_width=True, type="primary"):
            st.session_state['selected_game'] = game_data['name']
            st.rerun()

def get_top_games(df, n=10):
    """Lấy top N game nổi bật (dựa trên số đánh giá và tỉ lệ tích cực)"""
    df_temp = df.copy()
    
    # Kiểm tra xem các cột rating có tồn tại không
    if 'positive_ratings' in df_temp.columns and 'negative_ratings' in df_temp.columns:
        df_temp['total_ratings'] = df_temp['positive_ratings'] + df_temp['negative_ratings']
        df_temp['positive_ratio'] = df_temp['positive_ratings'] / (df_temp['total_ratings'] + 1)
        df_temp['score'] = df_temp['total_ratings'] * df_temp['positive_ratio']
        return df_temp.nlargest(n, 'score')
    else:
        # Fallback: trả về n game đầu tiên nếu không có rating
        return df_temp.head(n)

def get_new_releases(df, n=10):
    """Lấy game mới phát hành"""
    if 'release_date_parsed' in df.columns:
        df_valid = df[df['release_date_parsed'].notna()].copy()
        return df_valid.nlargest(n, 'release_date_parsed')
    return df.head(n)

def get_most_positive(df, n=10):
    """Lấy game có nhiều đánh giá tích cực nhất"""
    # Kiểm tra xem các cột rating có tồn tại không
    if 'positive_ratings' in df.columns and 'negative_ratings' in df.columns:
        df_temp = df[df['positive_ratings'] > 100].copy()
        df_temp['positive_ratio'] = df_temp['positive_ratings'] / (df_temp['positive_ratings'] + df_temp['negative_ratings'] + 1)
        return df_temp.nlargest(n, 'positive_ratio')
    else:
        # Fallback: trả về n game đầu tiên nếu không có rating
        return df.head(n)

def get_all_genres(df):
    """Lấy tất cả thể loại game"""
    all_genres = set()
    for genres in df['genres'].dropna():
        if ';' in str(genres):
            all_genres.update(genres.split(';'))
        else:
            all_genres.add(str(genres))
    return sorted(list(all_genres))

def get_recommendations(game_name, df, sim_matrix, top_k=9):
    """Lấy gợi ý game tương tự"""
    try:
        idx = df[df['name'] == game_name].index[0]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        rec_indices = [i[0] for i in sorted_scores]
        return df.iloc[rec_indices]
    except:
        return pd.DataFrame()

# 4. GIAO DIỆN CHÍNH
if df_games is not None:
    # Khởi tạo session state cho selected_game
    if 'selected_game' not in st.session_state:
        st.session_state['selected_game'] = None
    
    # HEADER
    st.title("🎮 STEAM GAME STORE")
    st.markdown("### Khám phá và tìm kiếm game yêu thích của bạn")
    
    # SIDEBAR - Bộ lọc
    with st.sidebar:
        st.markdown("## 🔍 Bộ lọc")
        
        # Filter theo thể loại
        all_genres = get_all_genres(df_games)
        selected_genre = st.selectbox(
            "Chọn thể loại game:",
            options=["Tất cả"] + all_genres,
            index=0
        )
        
        # Filter theo giá
        price_filter = st.radio(
            "Giá:",
            options=["Tất cả", "Miễn phí", "Có phí"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📊 Thống kê")
        st.metric("Tổng số game", len(df_games))
        free_games = len(df_games[df_games['price'] == 0])
        st.metric("Game miễn phí", free_games)
    
    # Áp dụng filter
    df_filtered = df_games.copy()
    
    if selected_genre != "Tất cả":
        df_filtered = df_filtered[df_filtered['genres'].str.contains(selected_genre, na=False)]
    
    if price_filter == "Miễn phí":
        df_filtered = df_filtered[df_filtered['price'] == 0]
    elif price_filter == "Có phí":
        df_filtered = df_filtered[df_filtered['price'] > 0]
    
    # TÌM KIẾM
    st.markdown("---")
    search_col1, search_col2 = st.columns([4, 1])
    
    with search_col1:
        # Lấy tất cả game từ df_games
        game_list = df_games['name'].tolist()
        
        # Tính toán index mặc định cho Selectbox
        default_index = 0
        if st.session_state.get('selected_game'):
            try:
                # Tìm vị trí của game trong danh sách (+1 vì có phần tử rỗng ở đầu)
                default_index = game_list.index(st.session_state['selected_game']) + 1
            except ValueError:
                default_index = 0
        
        # Tạo selectbox với index động
        search_query = st.selectbox(
            "🔎 Tìm kiếm game:",
            options=[""] + game_list,
            index=default_index,  # Set index theo game đã chọn
            placeholder="Nhập tên game để tìm kiếm..."
        )
        
        # Reset session state sau khi đã set index xong để không bị kẹt
        if st.session_state.get('selected_game'):
            st.session_state['selected_game'] = None
    
    with search_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Tìm kiếm", use_container_width=True)
    
    # NẾU CÓ TÌM KIẾM
    if search_query and search_query != "":
        st.markdown("---")
        st.markdown(f"## 🎯 Kết quả tìm kiếm: {search_query}")
        
        # Hiển thị game được tìm
        game_info = df_games[df_games['name'] == search_query].iloc[0]
        
        col_left, col_right = st.columns([2, 3])
        
        with col_left:
            st.image(game_info['header_image'], use_container_width=True)
            
            # Video trailer (nếu có)
            if 'movies' in game_info and pd.notna(game_info['movies']) and game_info['movies']:
                try:
                    st.markdown("### 🎬 Trailer")
                    movies_data = game_info['movies']
                    
                    if isinstance(movies_data, str):
                        try:
                            movies_data = ast.literal_eval(movies_data)
                        except:
                            movies_data = None
                    
                    if movies_data and len(movies_data) > 0:
                        video_url = None
                        
                        if isinstance(movies_data[0], dict):
                            if 'webm' in movies_data[0]:
                                video_url = movies_data[0]['webm'].get('max') or movies_data[0]['webm'].get('480')
                            elif 'mp4' in movies_data[0]:
                                video_url = movies_data[0]['mp4'].get('max') or movies_data[0]['mp4'].get('480')
                        
                        if video_url:
                            st.video(video_url)
                except Exception as e:
                    pass
        
        with col_right:
            st.markdown(f"## {game_info['name']}")
            
            # Mô tả
            if 'short_description' in game_info and pd.notna(game_info['short_description']):
                st.markdown("### 📝 Giới thiệu:")
                description = str(game_info['short_description'])
                if len(description) > 300:
                    description = description[:300] + "..."
                
                try:
                    if translator:
                        translated = translator.translate(description)
                        st.write(translated)
                    else:
                        st.write(description)
                except:
                    st.write(description)
            
            # Thông tin chi tiết
            st.markdown(f" Nhà phát triển: {game_info['developer']}")
            st.markdown(f" Ngày phát hành: {game_info.get('release_date', 'N/A')}")
            if game_info['price'] == 0:
                st.markdown(" Giá:  Miễn phí")
            else:
                st.markdown(f" Giá: ${game_info['price']}")
            genres = game_info['genres'].split(';') if ';' in str(game_info['genres']) else [game_info['genres']]
            st.markdown(f"** Thể loại:** {', '.join(genres[:3])}")
            
            # Tags
            if 'steamspy_tags' in game_info and pd.notna(game_info['steamspy_tags']):
                tags_str = str(game_info['steamspy_tags'])
                if tags_str and tags_str != 'nan':
                    # Xử lý tags (có thể là string hoặc dict)
                    try:
                        if isinstance(game_info['steamspy_tags'], str):
                            tags_dict = ast.literal_eval(tags_str)
                        else:
                            tags_dict = game_info['steamspy_tags']
                        
                        if isinstance(tags_dict, dict):
                            # Lấy top 5 tags phổ biến nhất
                            top_tags = sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)[:5]
                            tags_display = [tag[0] for tag in top_tags]
                            tags_html = ' '.join([f'<span style="background-color:#2a475e;padding:4px 10px;border-radius:3px;margin:2px;display:inline-block;font-size:12px;">🏷️ {tag}</span>' for tag in tags_display])
                            st.markdown(f"**Tags:** ", unsafe_allow_html=True)
                            st.markdown(tags_html, unsafe_allow_html=True)
                    except:
                        pass
            
            # Cấu hình yêu cầu
            if 'minimum' in game_info and pd.notna(game_info['minimum']):
                requirements = str(game_info['minimum'])
                if requirements and requirements != 'nan' and len(requirements) > 10:
                    st.markdown("**💻 Cấu hình yêu cầu:**")
                    
                    # Tạo expander để không chiếm quá nhiều không gian
                    with st.expander("Xem chi tiết cấu hình", expanded=False):
                        # Format requirements text
                        requirements_formatted = requirements.replace(', ', '\n• ')
                        if not requirements_formatted.startswith('•'):
                            requirements_formatted = '• ' + requirements_formatted
                        st.markdown(f"<div style='background-color:#16202d;padding:15px;border-radius:5px;font-size:13px;line-height:1.8;'>{requirements_formatted}</div>", unsafe_allow_html=True)
            
            if 'positive_ratings' in game_info and 'negative_ratings' in game_info:
                total = game_info['positive_ratings'] + game_info['negative_ratings']
                if total > 0:
                    positive_pct = (game_info['positive_ratings'] / total) * 100
                    st.markdown(f"**📊 Đánh giá: {positive_pct:.0f}% tích cực**")
                    st.progress(positive_pct / 100)
                    st.caption(f"👍 {game_info['positive_ratings']:,} | 👎 {game_info['negative_ratings']:,} đánh giá")
        
        # GAME GỢI Ý
        st.markdown("---")
        st.markdown("## 💡 Game tương tự bạn có thể thích")
        
        rec_games = get_recommendations(search_query, df_games, sim_matrix, top_k=9)
        
        if not rec_games.empty:
            # Hiển thị 3 hàng x 3 cột
            for row in range(3):
                cols = st.columns(3)
                for col_idx in range(3):
                    game_idx = row * 3 + col_idx
                    if game_idx < len(rec_games):
                        game = rec_games.iloc[game_idx]
                        with cols[col_idx]:
                            display_game_card(game)
        
    # TRANG CHỦ - Hiển thị các section
    else:
        st.markdown("---")
        
        # SECTION 1: TOP 10 GAME NỔI BẬT
        st.markdown('<p class="section-title">⭐ Top 10 Game Nổi Bật</p>', unsafe_allow_html=True)
        top_games = get_top_games(df_filtered, n=10)
        
        # Hiển thị 2 hàng x 5 cột
        for row in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                game_idx = row * 5 + col_idx
                if game_idx < len(top_games):
                    game = top_games.iloc[game_idx]
                    with cols[col_idx]:
                        display_game_card(game)
        
        st.markdown("---")
        
        # SECTION 2: GAME MỚI PHÁT HÀNH
        st.markdown('<p class="section-title">🆕 Game Mới Phát Hành</p>', unsafe_allow_html=True)
        new_games = get_new_releases(df_filtered, n=10)
        
        for row in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                game_idx = row * 5 + col_idx
                if game_idx < len(new_games):
                    game = new_games.iloc[game_idx]
                    with cols[col_idx]:
                        display_game_card(game)
        
        st.markdown("---")
        
        # SECTION 3: GAME CÓ NHIỀU ĐÁNH GIÁ TÍCH CỰC
        st.markdown('<p class="section-title">👍 Game Được Đánh Giá Cao</p>', unsafe_allow_html=True)
        positive_games = get_most_positive(df_filtered, n=10)
        
        for row in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                game_idx = row * 5 + col_idx
                if game_idx < len(positive_games):
                    game = positive_games.iloc[game_idx]
                    with cols[col_idx]:
                        display_game_card(game)

else:
    st.error("❌ Không thể load dữ liệu. Vui lòng kiểm tra lại!")
