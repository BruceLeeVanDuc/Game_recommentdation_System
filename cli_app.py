import argparse
import pandas as pd
import pickle
import sys

def load_resources():
    print("⏳ Loading model resources...")
    try:
        with open('models/hybrid_similarity.pkl', 'rb') as f:
            sim_matrix = pickle.load(f)
        
        df_games = pd.read_pickle('models/games_metadata.pkl')
        print("✅ Resources loaded successfully.")
        return df_games, sim_matrix
    except FileNotFoundError:
        print("❌ Error: Model files not found. Please run train_model.py first.")
        sys.exit(1)

def get_recommendations(game_name, df, sim_matrix, top_k=10):
    # Case-insensitive search
    matches = df[df['name'].str.lower() == game_name.lower()]
    
    if len(matches) == 0:
        # Fuzzy search alternative
        print(f"⚠️ Game '{game_name}' not found exactly.")
        print("   Did you mean one of these?")
        # Simple contains search
        candidates = df[df['name'].str.lower().str.contains(game_name.lower())].head(5)
        for _, row in candidates.iterrows():
            print(f"   - {row['name']}")
        return []

    idx = matches.index[0]
    real_name = matches.iloc[0]['name']
    print(f"🔎 Finding recommendations for: {real_name}")
    
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
    
    results = []
    for i, (game_idx, score) in enumerate(sim_scores):
        game_row = df.iloc[game_idx]
        results.append({
            'Rank': i + 1,
            'Name': game_row['name'],
            'Similarity': f"{score:.4f}",
            'Genres': game_row['genres'],
            'Price': game_row['price']
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Steam Game Recommender CLI")
    parser.add_argument('game', type=str, nargs='?', help="Name of the game to find recommendations for")
    parser.add_argument('--top', type=int, default=10, help="Number of recommendations returned")
    parser.add_argument('--list', action='store_true', help="List random available games")
    
    args = parser.parse_args()
    
    df_games, sim_matrix = load_resources()

    if args.list:
        print("\n🎲 Random 10 games available in dataset:")
        print(df_games['name'].sample(10).to_string(index=False))
        return

    if not args.game:
        # Interactive mode if no arg provided
        while True:
            query = input("\n🎮 Enter game name (or 'q' to quit): ").strip()
            if query.lower() in ['q', 'quit', 'exit']:
                break
            if not query:
                continue
                
            results = get_recommendations(query, df_games, sim_matrix, args.top)
            if results:
                print("\n🏆 Top Recommendations:")
                print(pd.DataFrame(results).to_string(index=False))
    else:
        # Direct mode
        results = get_recommendations(args.game, df_games, sim_matrix, args.top)
        if results:
            print("\n🏆 Top Recommendations:")
            print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
