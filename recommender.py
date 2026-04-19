import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- AYARLAR ---
WEIGHT_NLP = 0.5      # İçerik/Hikaye benzerliği
WEIGHT_GENRE = 0.3    # Tür uyumu
WEIGHT_RATING = 0.2   # IMDB/TMDB puanı
DIRECTOR_BONUS = 0.1  # Aynı yönetmense eklenecek bonus puan

def create_soup(x):
    
    
    
    soup_parts = []
    
    if pd.notna(x['overview']):
        soup_parts.append(x['overview'])
    
    
    if pd.notna(x['genres']):
        soup_parts.append(x['genres'].replace('|', ' '))
        
    
    if 'keywords' in x and pd.notna(x['keywords']):
         soup_parts.append(x['keywords'])

    return ' '.join(soup_parts)

def load_data():
    print("Veriler yükleniyor...")
    
    metadata = pd.read_csv('data/tmdb_movies_metadata.csv', low_memory=False)
    movies = pd.read_csv('data/movies.csv', low_memory=False)
    
    
    data = pd.merge(metadata, movies, on='movieId')
    
    #  OY SAYISI FİLTRESİ (Gürültü Temizliği) ---
    # 100'den az oy almış filmleri baştan eliyoruz.
   
    if 'vote_count' in data.columns:
        print(f"Filtreleme öncesi film sayısı: {len(data)}")
        data = data[data['vote_count'] >= 100]
        print(f"Filtreleme sonrası (Vote > 100) film sayısı: {len(data)}")
    
    data = data.dropna(subset=['overview'])
    
    # Puan normalizasyonu (0-1 arasına çekiyoruz)
    data['normalized_rating'] = data['tmdb_vote_average'] / 10.0
    
    # Türleri set (küme) haline getiriyoruz
    data['genres_set'] = data['genres'].apply(lambda x: set(x.split('|')) if pd.notna(x) else set())
    
    # Soup sütununu oluşturuyoruz
    data['soup'] = data.apply(create_soup, axis=1)
    
    
    data = data.reset_index(drop=True)
    
    return data

def create_similarity_matrix(data):
    #  Artık 'overview' yerine 'soup' kullanıyoruz.
    tfidf = TfidfVectorizer(stop_words='english')
    
    tfidf_matrix = tfidf.fit_transform(data['soup'])
    
    print("Matris hesaplanıyor (Bu işlem biraz sürebilir)...")
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def calculate_genre_score(set1, set2):
    # Jaccard Benzerliği: Kesişim / Birleşim
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def get_recommendations(title, data, cosine_sim, indices):
    try:
        idx = indices[title]
    
        if type(idx) == pd.Series:
            idx = idx.iloc[0]
    except KeyError:
        return None

    
    target_genres = data.iloc[idx]['genres_set']
    
    target_director = data.iloc[idx]['director'] if 'director' in data.columns else None

    # --- KATI KURALLAR (HARD FILTERS) ---
    is_kids_movie = 'Children' in target_genres or 'Animation' in target_genres

    sim_scores = list(enumerate(cosine_sim[idx]))
    
    
    sim_scores = [x for x in sim_scores if x[0] != idx]

    final_scores = []
    
    for i, nlp_score in sim_scores:
        # Hızlandırma: Sadece NLP benzerliği %5 üzeri olanlara bak
        if nlp_score > 0.05:
            candidate_row = data.iloc[i]
            candidate_genres = candidate_row['genres_set']
            
            # --- FİLTRE: Çocuk Filmi Koruması ---
            if is_kids_movie:
                if not ('Children' in candidate_genres or 'Animation' in candidate_genres):
                    continue
            
            
            
            # 1. Puan Skoru
            rating_score = candidate_row['normalized_rating']
            
            # 2. Tür Skoru
            genre_score = calculate_genre_score(target_genres, candidate_genres)
            
            # 3. Hibrit Hesaplama
            hybrid_score = (nlp_score * WEIGHT_NLP) + \
                           (genre_score * WEIGHT_GENRE) + \
                           (rating_score * WEIGHT_RATING)
            
            # --- MADDE 5: YÖNETMEN BONUSU ---
            # Eğer veri setinde yönetmen bilgisi varsa ve aynıysa puan ekle
            if target_director and 'director' in candidate_row:
                if candidate_row['director'] == target_director:
                    hybrid_score += DIRECTOR_BONUS

            final_scores.append((i, hybrid_score))

    # Puanı en yüksekten düşüğe sırala
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    final_scores = final_scores[:10]
    
    movie_indices = [i[0] for i in final_scores]
    
    # Sonuç sütunlarını seç
    cols_to_show = ['title', 'genres', 'tmdb_vote_average']
    if 'vote_count' in data.columns:
        cols_to_show.append('vote_count')
        
    return data[cols_to_show].iloc[movie_indices]

def main():
    data = load_data()
    
    # İndex serisini oluştur
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    
    # Matrisi oluştur
    cosine_sim = create_similarity_matrix(data)
    
    print("-" * 50)
    print("SİSTEM HAZIR (v3.5 - Hibrit + Vote Filter + Soup)")
    print("-" * 50)
    
    while True:
        user_input = input("\nFilm adı (Çıkış için 'q'): ")
        if user_input.lower() == 'q':
            break
            
        recommendations = get_recommendations(user_input, data, cosine_sim, indices)
        
        if recommendations is not None:
            print(f"\n'{user_input}' için ÖNERİLER:")
            print("-" * 80)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            print(recommendations.to_string(index=False))
        else:
            print("Film bulunamadı! Tam ismini doğru yazdığından emin ol.")

if __name__ == "__main__":
    main()