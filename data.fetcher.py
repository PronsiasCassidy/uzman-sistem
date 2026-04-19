import pandas as pd
import requests
import time
import os

# --- AYARLAR ---
API_KEY = 'ac1d82dafefedbd48263bc8c38cf8e2d' 

INPUT_PATH = 'data/links.csv'
OUTPUT_PATH = 'data/tmdb_movies_metadata.csv'

def get_movie_details(tmdb_id):
    """TMDB API'sinden film detaylarını (Özet ve Puan) çeker."""
   
    if pd.isna(tmdb_id):
        return None, None
    
    
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={API_KEY}&language=en-US"
    
    try:
        response = requests.get(url, timeout=5)
        
        
        if response.status_code == 200:
            data = response.json()
            overview = data.get('overview', '')
            vote_average = data.get('vote_average', 0)
            return overview, vote_average
            
        
        elif response.status_code == 429:
            print("⚠️ Hız sınırı! 5 saniye bekleniyor...")
            time.sleep(5)
            return get_movie_details(tmdb_id)
            
        else:
            return None, None
            
    except Exception as e:
        print(f"Hata (ID: {tmdb_id}): {e}")
        return None, None

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"HATA: {INPUT_PATH} dosyası bulunamadı! 'data' klasörünü kontrol et.")
        return

    print("--- Veri Çekme İşlemi Başlıyor ---")
    
    
    links_df = pd.read_csv(INPUT_PATH, dtype={'tmdbId': 'Int64'})
    
   
    
    total_movies = len(links_df)
    print(f"Toplam {total_movies} film taranacak.")

    overviews = []
    tmdb_ratings = []
    
    print("İndirme başladı...")
    
    for index, row in links_df.iterrows():
        tmdb_id = row['tmdbId']
        
        overview, rating = get_movie_details(tmdb_id)
        
        overviews.append(overview)
        tmdb_ratings.append(rating)
        
        
        print(f"[{index+1}/{total_movies}] ID: {tmdb_id} - Tamamlandı")
        
        
        time.sleep(0.2)

    
    links_df['overview'] = overviews
    links_df['tmdb_vote_average'] = tmdb_ratings
    
    
    final_df = links_df.dropna(subset=['overview'])
    final_df = final_df[final_df['overview'] != '']
    
    
    final_df.to_csv(OUTPUT_PATH, index=False)
    print("-" * 30)
    print(f"✅ İŞLEM TAMAM! Dosya kaydedildi: {OUTPUT_PATH}")
    print(f"Toplam {len(final_df)} adet temiz film verisi elde edildi.")

if __name__ == "__main__":
    main()