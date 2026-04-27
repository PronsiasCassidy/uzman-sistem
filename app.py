import os
import json
import math
import csv
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

LOG_FILE = Path("logs/anon_results.csv")

def save_anonymous_log(answers, scores, result):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "answers": json.dumps(answers, ensure_ascii=False),
        "uyum": scores[0],
        "ahlak": scores[1],
        "varolus": scores[2],
        "karar": scores[3],
        "archetype": result.get("name"),
        "character": result.get("closest_character"),
    }

    file_exists = LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

app.json_encoder = NpEncoder

TMDB_API_KEY = 'ac1d82dafefedbd48263bc8c38cf8e2d'
MIN_VOTE_COUNT = 100
WEIGHT_NLP = 0.5
WEIGHT_RATING = 0.2
WEIGHT_GENRE = 0.3

GENRE_MAP = {
    'Aksiyon': 28, 'Macera': 12, 'Animasyon': 16, 'Komedi': 35, 'Suç': 80,
    'Dram': 18, 'Fantastik': 14, 'Korku': 27, 'Gizem': 9648, 'Romantik': 10749,
    'Bilim Kurgu': 878, 'Gerilim': 53
}

movies = pd.DataFrame()
model_knn = None
movie_user_matrix = None
tfidf_matrix = None
nlp_indices = {}
movielens_to_tmdb = {}
tmdb_to_movielens = {}
movie_id_to_matrix_idx = {}
matrix_idx_to_movie_id = {}
title_to_tmdb_id = {}

def load_data_safe(filename, low_memory=False):
    if os.path.exists(filename):
        return pd.read_csv(filename, low_memory=low_memory)
    path = os.path.join('data', filename)
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=low_memory)
    return pd.DataFrame()

def calculate_genre_score(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def get_tmdb_data(tmdb_id, endpoint=""):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}{endpoint}?api_key={TMDB_API_KEY}&language=tr-TR"
    try:
        r = requests.get(url, timeout=2)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def get_poster(tmdb_id, path=None, size="w500"):
    if path and pd.notna(path) and str(path).lower() != 'nan':
        clean_path = str(path).strip()
        if not clean_path.startswith('/'):
            clean_path = '/' + clean_path
        return f"https://image.tmdb.org/t/p/{size}{clean_path}"

    if tmdb_id:
        try:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
            res = requests.get(url, timeout=1).json()
            api_path = res.get('poster_path')
            if api_path:
                return f"https://image.tmdb.org/t/p/{size}{api_path}"
        except Exception:
            pass

    return "https://via.placeholder.com/300x450?text=No+Poster"

_tmdb_multi_cache = {}

def search_tmdb_multi(query, lang="tr-TR"):
    if not query:
        return []
    cache_key = f"{query}_{lang}"
    if cache_key in _tmdb_multi_cache:
        return _tmdb_multi_cache[cache_key]

    url = "https://api.themoviedb.org/3/search/multi"
    params = {"api_key": TMDB_API_KEY, "language": lang, "query": query}
    try:
        r = requests.get(url, params=params, timeout=3)
        results = r.json().get("results", []) if r.status_code == 200 else []
        _tmdb_multi_cache[cache_key] = results
        return results
    except Exception:
        return []

def get_work_match_from_title(title, preferred_media_type=None):
    if not title:
        return None

    import re
    # Parantez içindeki açıklamaları temizle örn: "The Matrix (1999)" -> "The Matrix"
    clean_title = re.sub(r'\(.*?\)', '', title).strip()
    
    # "Season 1", ": Sezon 2", "- Part 3" gibi dizi alt başlıklarını temizle
    clean_title = re.sub(r'(:|-)?\s*(Season|Sezon|Volume|Vol\.?|Part)\s*\d+.*$', '', clean_title, flags=re.IGNORECASE).strip()

    # 1. Aşama: Türkçe Ara
    results = search_tmdb_multi(clean_title, lang="tr-TR")
    
    # 2. Aşama: Türkçe bulamazsa İngilizce Ara
    if not results:
        results = search_tmdb_multi(clean_title, lang="en-US")
        
    # 3. Aşama: Hala bulamazsa orijinal temizlenmemiş isimle Ara (İngilizce)
    if not results and clean_title != title:
        results = search_tmdb_multi(title, lang="en-US")

    filtered = []
    for item in results:
        media_type = item.get("media_type")
        if media_type not in ["movie", "tv"]:
            continue
        if preferred_media_type and media_type != preferred_media_type:
            continue
        filtered.append(item)

    candidates = filtered if filtered else [x for x in results if x.get("media_type") in ["movie", "tv"]]

    if not candidates:
        return None

    # Posteri olan sonuçlara öncelik ver
    candidates_with_poster = [c for c in candidates if c.get("poster_path")]
    best = candidates_with_poster[0] if candidates_with_poster else candidates[0]
    
    poster_path = best.get("poster_path")
    return {
        "poster": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
        "tmdb_id": best.get("id"),
        "media_type": best.get("media_type"),
        "title": best.get("title") or best.get("name")
    }

SCENARIO_VECTORS = {
    # [uyum, ahlak, varolus, karar]

    # 1) Şirket yolsuzluğu  
    1: {
        'A': [22, 10, 3, -5],   
        'B': [-22, 0, 0, 5],    
        'C': [-10, -18, 0, 10]  
    },

    # 2) Kural esnetme  → iyi
    2: {
        'A': [20, 8, 0, -4],    
        'B': [-20, 0, 0, 4],    
        'C': [-10, -10, 0, 8]   
    },

    # 3) %20 yaşama şansı 
    3: {
        'A': [0, 0, 20, 5],     
        'B': [0, 0, -20, 5],    
        'C': [0, -10, -12, -5]  
    },

    # 4) Projenin çöpe atılması 
    4: {
        'A': [0, 0, 16, 6],     
        'B': [0, 0, -16, 2],    
        'C': [0, -8, -6, 4]     
    },

    # 5) Salgın 
    5: {
        'A': [0, -22, 0, 10],   
        'B': [0, 28, 0, 0],     
        'C': [0, -10, 0, 5]     
    },

    # 6) Not paylaşma → iyi
    6: {
        'A': [0, 12, 0, 0],     
        'B': [0, -10, 0, 2],
        'C': [0, -4, 0, 6]
    },

    # 7) Ütopik yalan 
    7: {
        'A': [0, 10, 0, -18],   
        'B': [0, -8, 0, 18],    
        'C': [0, 4, -6, 10]     
    },

    # 8) İntikam 
    8: {
        'A': [0, 12, 3, 14],    
        'B': [0, -12, 0, -16],
        'C': [0, -4, 0, 10]
    }
}

ARCHETYPE_META = {
    "Özgeci / Uyumlu / İradeci / Mantıksal": {
        "name": "Stratejik Koruyucu",
        "desc": "İlkelerini korurken düzeni tamamen yıkmadan hareket eden, uzun vadeyi gören koruyucu zihin.",
        "img": "https://via.placeholder.com/150/1d4ed8/ffffff?text=Koruyucu"
    },
    "Özgeci / Uyumlu / İradeci / Duygusal": {
        "name": "Duygusal Koruyucu",
        "desc": "Bağlılık, vicdan ve sadakatle hareket eden; doğru bildiğini kalbiyle savunan koruyucu profil.",
        "img": "https://via.placeholder.com/150/2563eb/ffffff?text=Sadik"
    },
    "Özgeci / Uyumlu / Determinist / Mantıksal": {
        "name": "Analitik Gözlemci",
        "desc": "Olayların akışını dışarıdan okuyup duygudan çok yapıyı gören, soğukkanlı özgeci gözlemci.",
        "img": "https://via.placeholder.com/150/0f766e/ffffff?text=Gozlemci"
    },
    "Özgeci / Uyumlu / Determinist / Duygusal": {
        "name": "Melankolik Gözlemci",
        "desc": "Kader ve kırılganlık duygusunu derinden hisseden, pasif ama vicdanlı gözlemci ruh.",
        "img": "https://via.placeholder.com/150/134e4a/ffffff?text=Melankoli"
    },
    "Özgeci / Kaotik / İradeci / Mantıksal": {
        "name": "Hesaplı Reformist",
        "desc": "Düzeni kırmaktan çekinmeyen ama bunu öfkeyle değil, stratejik akılla yapan dönüştürücü profil.",
        "img": "https://via.placeholder.com/150/7c3aed/ffffff?text=Reform"
    },
    "Özgeci / Kaotik / İradeci / Duygusal": {
        "name": "Tutkulu Reformist",
        "desc": "Adaletsizlik karşısında beklemeyen, dünyayı değiştirmeyi duygusal enerjiyle isteyen idealist asi.",
        "img": "https://via.placeholder.com/150/9333ea/ffffff?text=Tutkulu"
    },
    "Özgeci / Kaotik / Determinist / Mantıksal": {
        "name": "Nihilist Asi",
        "desc": "Düzenin çürümüşlüğünü gören, umudu zayıf ama zekâsı güçlü, soğuk başkaldıran profil.",
        "img": "https://via.placeholder.com/150/6d28d9/ffffff?text=Asi"
    },
    "Özgeci / Kaotik / Determinist / Duygusal": {
        "name": "Tepkisel Asi",
        "desc": "Sistemle barışmayan, kırgınlığını ve isyanını bastırmayan, dürtüsel başkaldıran karakter.",
        "img": "https://via.placeholder.com/150/a21caf/ffffff?text=Tepki"
    },
    "Bencil / Uyumlu / İradeci / Mantıksal": {
        "name": "Sistemik Oportünist",
        "desc": "Kuralları yıkmak yerine onları kendi lehine bükmeyi bilen, kontrollü güç oyuncusu.",
        "img": "https://via.placeholder.com/150/92400e/ffffff?text=Oportunist"
    },
    "Bencil / Uyumlu / İradeci / Duygusal": {
        "name": "Kibirli Otokrat",
        "desc": "Sistemin içinde kalıp üstünlük kurmak isteyen, egosu ve kontrol arzusu yüksek profil.",
        "img": "https://via.placeholder.com/150/b45309/ffffff?text=Otokrat"
    },
    "Bencil / Uyumlu / Determinist / Mantıksal": {
        "name": "Sessiz Pragmatist",
        "desc": "Büyük idealler yerine kendi güvenliğini ve çıkarını koruyan, sessiz ama hesaplı pragmatist.",
        "img": "https://via.placeholder.com/150/57534e/ffffff?text=Pragmatik"
    },
    "Bencil / Uyumlu / Determinist / Duygusal": {
        "name": "Kaçak Pragmatist",
        "desc": "Yüksek riskten kaçan, sorumluluğu üstlenmeyen ama fırsat bulduğunda kendi lehine kayan profil.",
        "img": "https://via.placeholder.com/150/78716c/ffffff?text=Kacak"
    },
    "Bencil / Kaotik / İradeci / Mantıksal": {
        "name": "Vizyoner Diktatör",
        "desc": "Amaca ulaşmak için sert bedeller ödetmeyi meşru gören, büyük ölçekli ve stratejik güç profili.",
        "img": "https://via.placeholder.com/150/991b1b/ffffff?text=Diktator"
    },
    "Bencil / Kaotik / İradeci / Duygusal": {
        "name": "Öfkeli Fatih",
        "desc": "Kontrol ve üstünlük arzusunu doğrudan eyleme döken, saldırgan ve dışavurumcu güç karakteri.",
        "img": "https://via.placeholder.com/150/b91c1c/ffffff?text=Fatih"
    },
    "Bencil / Kaotik / Determinist / Mantıksal": {
        "name": "Soğukkanlı Yıkıcı",
        "desc": "Dünyaya duygusal bağlılığı zayıf, zarar vermeyi araç değil sonuç olarak da kabul edebilen buz gibi profil.",
        "img": "https://via.placeholder.com/150/111827/ffffff?text=Yikici"
    },
    "Bencil / Kaotik / Determinist / Duygusal": {
        "name": "Saf Dürtüsel Yıkıcı",
        "desc": "Öfke, boşluk ve dürtüyle hareket eden; sonuçtan çok patlamanın kendisini yaşayan yıkıcı karakter.",
        "img": "https://via.placeholder.com/150/1f2937/ffffff?text=Durtusel"
    },
}

class UserProfile:
    def __init__(self):
        self.scores = [50, 50, 50, 50]

    def apply_vector(self, vector):
        for i in range(4):
            self.scores[i] += vector[i]

    def finalize_scores(self):
        for i in range(4):
            self.scores[i] = max(0, min(100, self.scores[i]))

    def get_normalized(self):
        return [(s - 50) / 50.0 for s in self.scores]

class ExpertEngine:
    def __init__(self, csv_path='data/character_axis_profiles.csv', codebook_path='templates/codebook.html'):
        self.weights = [1.2, 1.2, 1.4, 1.0]
        self.characters = self._load_character_profiles(csv_path, codebook_path)

    def _load_character_profiles(self, csv_path, codebook_path):
        if not os.path.exists(csv_path):
            print("❌ character_axis_profiles.csv bulunamadı")
            return pd.DataFrame()

        df = pd.read_csv(csv_path)
        print(f"✅ character_axis_profiles yüklendi: {len(df)} satır")

        codebook_candidates = [
            codebook_path,
            os.path.join('templates', 'codebook.html'),
            os.path.join('data', 'codebook.html'),
        ]
        codebook_file = next((p for p in codebook_candidates if os.path.exists(p)), None)

        if not codebook_file:
            print("❌ codebook dosyası bulunamadı")
            return df

        try:
            print(f"✅ Codebook bulundu: {codebook_file}")
            tables = pd.read_html(codebook_file)
            print(f"✅ Okunan tablo sayısı: {len(tables)}")

            selected = None
            for i, tbl in enumerate(tables):
                cols = [str(c).strip().lower() for c in tbl.columns]
                joined = " | ".join(cols)
                if "character display name" in joined and "fictional work" in joined:
                    selected = tbl.copy()
                    print(f"✅ Doğru tablo bulundu: {i}")
                    break

            if selected is None:
                print("❌ Uygun codebook tablosu bulunamadı")
                return df

            selected = selected.iloc[:, :3].copy()
            selected.columns = ['codebook_id', 'character_name', 'fictional_work']
            selected['character_id'] = range(len(selected))

            df = df.merge(
                selected[['character_id', 'character_name', 'fictional_work']],
                on='character_id',
                how='left'
            )
            filled = int(df['character_name'].notna().sum())
            print(f"✅ Merge tamamlandı. character_name dolu satır: {filled}")

        except Exception as e:
            print(f"❌ Codebook okuma hatası: {e}")

        return df
    
   

    def infer_archetype(self, scores):
        uyum, ahlak, varolus, karar = scores
        ahlak_side = 'Özgeci' if ahlak >= 50 else 'Bencil'
        uyum_side = 'Kaotik' if uyum >= 50 else 'Uyumlu'
        varolus_side = 'İradeci' if varolus >= 50 else 'Determinist'
        karar_side = 'Mantıksal' if karar >= 50 else 'Duygusal'

        label = f"{ahlak_side} / {uyum_side} / {varolus_side} / {karar_side}"
        meta = ARCHETYPE_META.get(label, {
            'name': label,
            'desc': 'Bu profil için açıklama bulunamadı.',
            'img': 'https://via.placeholder.com/150/444/fff?text=Arketip'
        })

        return {'label': label, 'name': meta['name'], 'desc': meta['desc'], 'img': meta['img']}

    def weighted_distance(self, user_scores, char_scores):
        return math.sqrt(sum(
            self.weights[i] * (user_scores[i] - char_scores[i]) ** 2
            for i in range(4)
        ))

    def get_top_matches(self, user, top_n=30):
        if self.characters.empty:
            return []

        results = []
        for _, row in self.characters.iterrows():
            char_scores = [
                float(row['uyum']),
                float(row['ahlak']),
                float(row['varolus']),
                float(row['karar']),
            ]
            dist = self.weighted_distance(user.scores, char_scores)
            results.append({
                'distance': float(dist),
                'character_name': str(row['character_name']) if 'character_name' in row.index and pd.notna(row['character_name']) else f"Character #{int(row['character_id'])}",
                'source_title': str(row['fictional_work']) if 'fictional_work' in row.index and pd.notna(row['fictional_work']) else None,
                'scores': char_scores,
            })

        results.sort(key=lambda x: x['distance'])
        return results[:top_n]

    def get_closest(self, user):
        archetype = self.infer_archetype(user.scores)
        matches = self.get_top_matches(user, top_n=1)

        if not matches:
            return {
                'name': archetype['name'],
                'desc': archetype['desc'],
                'img': archetype['img'],
                'closest_character': None,
                'source_title': None,
                'distance': None
            }

        best = matches[0]
        desc = archetype['desc']
        if best.get('character_name'):
            if best.get('source_title'):
                desc += f" En yakın eşleşen karakter: {best['character_name']} ({best['source_title']})."
            else:
                desc += f" En yakın eşleşen karakter: {best['character_name']}."

        return {
            'name': archetype['name'],
            'desc': desc,
            'img': archetype['img'],
            'closest_character': best.get('character_name'),
            'source_title': best.get('source_title'),
            'distance': round(best.get('distance'), 3) if best.get('distance') is not None else None
        }

print("🚀 HİBRİT SİSTEM (v5.0 - DUAL MATCH + ANON LOG) Başlatılıyor...")

try:
    metadata = load_data_safe('tmdb_movies_metadata.csv', low_memory=False)
    links = load_data_safe('links.csv', low_memory=False)
    movies_orig = load_data_safe('movies.csv', low_memory=False)
    ratings = load_data_safe('ratings.csv', low_memory=False)

    if not metadata.empty:
        metadata.columns = [c.strip() for c in metadata.columns]
        if 'tmdb_vote_average' in metadata.columns:
            metadata.rename(columns={'tmdb_vote_average': 'vote_average'}, inplace=True)
        if 'tmdb_vote_count' in metadata.columns:
            metadata.rename(columns={'tmdb_vote_count': 'vote_count'}, inplace=True)
        if 'poster_path' not in metadata.columns:
            metadata['poster_path'] = None

    links = links.dropna(subset=['tmdbId', 'movieId'])
    links['tmdbId'] = links['tmdbId'].astype(int)
    links['movieId'] = links['movieId'].astype(int)
    movielens_to_tmdb = dict(zip(links.movieId, links.tmdbId))
    tmdb_to_movielens = dict(zip(links.tmdbId, links.movieId))

    movies = movies_orig[movies_orig['movieId'].isin(links['movieId'])].copy()
    movies['tmdbId'] = movies['movieId'].map(movielens_to_tmdb).fillna(0).astype(int)

    if not metadata.empty:
        meta_subset = metadata[['tmdbId', 'overview', 'vote_average', 'poster_path']].drop_duplicates('tmdbId')
        movies = movies.merge(meta_subset, on='tmdbId', how='left')
        movies['vote_average'] = movies['vote_average'].fillna(0)

    if not ratings.empty:
        valid_movie_ids = set(movies['movieId'])
        filt_ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]
        counts = filt_ratings['movieId'].value_counts()
        popular = counts[counts >= 20].index
        filt_ratings = filt_ratings[filt_ratings['movieId'].isin(popular)]

        movie_user_matrix = filt_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        movie_user_matrix_sparse = csr_matrix(movie_user_matrix.values)
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
        model_knn.fit(movie_user_matrix_sparse)

        movie_id_to_matrix_idx = {mid: i for i, mid in enumerate(movie_user_matrix.index)}
        matrix_idx_to_movie_id = {i: mid for i, mid in enumerate(movie_user_matrix.index)}
        print("✅ CF Modeli Hazır.")

    if not movies.empty:
        movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False)
        movies['overview'] = movies['overview'].fillna('')
        movies['soup'] = movies['genres_str'] + " " + movies['overview'] + " " + movies['genres_str']
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies['soup'])
        nlp_indices = pd.Series(movies.index, index=movies['movieId']).to_dict()
        print("✅ NLP Modeli Hazır.")

    for _, row in movies.iterrows():
        clean_t = str(row['title']).split('(')[0].strip().lower()
        title_to_tmdb_id[clean_t] = int(row['tmdbId'])

except Exception as e:
    print(f"⚠️ Veri yükleme hatası: {e}")

expert_engine = ExpertEngine(codebook_path='templates/codebook.html')

def get_match_level(distance):
    if distance < 15:
        return "Çok güçlü eşleşme"
    elif distance < 25:
        return "İyi eşleşme"
    elif distance < 40:
        return "Orta eşleşme"
    else:
        return "Zayıf eşleşme"


def get_similarity_percentage(distance):
    similarity = max(0, 100 - distance)
    return round(similarity, 1)

def generate_dynamic_comment(scores):
    uyum, ahlak, varolus, karar = scores
    fragments = []

    if uyum >= 70:
        fragments.append("sisteme karşı duran ve düzeni sorgulayan")
    elif uyum <= 30:
        fragments.append("düzeni korumayı ve yapıyı sürdürmeyi tercih eden")
    else:
        fragments.append("düzen ile başkaldırı arasında dengede duran")

    if ahlak >= 70:
        fragments.append("yüksek etik duyarlılığa sahip")
    elif ahlak <= 30:
        fragments.append("amacı için sert bedelleri kabul edebilen")
    else:
        fragments.append("ahlaki gri alanlarda karar veren")

    if varolus >= 60:
        fragments.append("iradesiyle anlam yaratmaya çalışan")
    elif varolus <= 40:
        fragments.append("dünyaya daha mesafeli ve gerçekçi yaklaşan")
    else:
        fragments.append("umut ile kabulleniş arasında gidip gelen")

    if karar >= 70:
        fragments.append("soğukkanlı ve stratejik")
    elif karar <= 30:
        fragments.append("anlık ve duygusal tepkiler verebilen")
    else:
        fragments.append("hem sezgi hem akılla hareket eden")

    return "Sen, " + ", ".join(fragments) + " bir profilsin."

def pick_dual_matches(top_matches):
    film_match = None
    series_match = None

    for match in top_matches:
        source_title = match.get('source_title')
        if not source_title:
            continue

        tmdb_info = get_work_match_from_title(source_title)
        if not tmdb_info:
            continue

        enriched = {
            "character_name": match.get("character_name"),
            "source_title": match.get("source_title"),
            "distance": round(match.get("distance", 0), 3),
            "tmdb": tmdb_info,
        }

        if tmdb_info.get("media_type") == "movie" and film_match is None:
            film_match = enriched
        elif tmdb_info.get("media_type") == "tv" and series_match is None:
            series_match = enriched

        if film_match and series_match:
            break

    return film_match, series_match

@app.route('/')
def index():
    active_mode = request.args.get('mode', 'live')
    slider_data = []
    grid_movies = []

    if active_mode == 'live':
        try:
            url = f"https://api.themoviedb.org/3/movie/now_playing?api_key={TMDB_API_KEY}&language=tr-TR"
            res = requests.get(url, timeout=3).json().get('results', [])
            for m in res:
                movie_item = {
                    'title': m['title'],
                    'vote_average': "{:.1f}".format(float(m.get('vote_average', 0))),
                    'backdrop_url': f"https://image.tmdb.org/t/p/original{m['backdrop_path']}" if m.get('backdrop_path') else f"https://image.tmdb.org/t/p/w500{m.get('poster_path')}",
                    'poster_url': f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else None,
                    'tmdbId': m['id']
                }
                grid_movies.append(movie_item)
                if len(slider_data) < 10:
                    slider_data.append(movie_item)
        except Exception as e:
            print(f"Vizyon hatası: {e}")
    else:
        if not movies.empty:
            sample_movies = movies.sort_values(by='vote_average', ascending=False).head(100)
            if len(sample_movies) >= 24:
                sample_movies = sample_movies.sample(24)

            for _, row in sample_movies.iterrows():
                p_url = get_poster(row['tmdbId'], row.get('poster_path'), size="w500")
                b_url = get_poster(row['tmdbId'], row.get('poster_path'), size="original")
                movie_item = {
                    'title': row['title'],
                    'vote_average': "{:.1f}".format(float(row.get('vote_average', 0))),
                    'poster_url': p_url,
                    'backdrop_url': b_url,
                    'tmdbId': int(row['tmdbId'])
                }
                grid_movies.append(movie_item)
                if len(slider_data) < 10:
                    slider_data.append(movie_item)

    return render_template(
        'index.html',
        slider_movies=slider_data,
        grid_movies=grid_movies,
        active_mode=active_mode,
        genres=GENRE_MAP.keys()
    )

@app.route('/suggest')
def suggest():
    q = request.args.get('q', '').lower()
    res = []
    if not movies.empty and q:
        matches = movies[movies['title'].str.lower().str.contains(q, na=False)].head(8)
        for _, row in matches.iterrows():
            p_url = get_poster(row.get('tmdbId'), row.get('poster_path'), size="w92")
            res.append({
                'title': str(row['title']),
                'tmdbId': int(row['tmdbId']),
                'poster': p_url
            })
    return jsonify(res)

@app.route('/get_movie_details', methods=['POST'])
def get_movie_details():
    tid = request.json.get('tmdbId')
    details = get_tmdb_data(tid)
    credits_data = get_tmdb_data(tid, "/credits")
    rec_data = get_tmdb_data(tid, "/recommendations")

    cast = [
        {
            'name': c['name'],
            'photo': f"https://image.tmdb.org/t/p/w200{c['profile_path']}" if c.get('profile_path') else ""
        }
        for c in credits_data.get('cast', [])[:6]
    ]
    recs = [
        {
            'title': r['title'],
            'tmdbId': r['id'],
            'poster': get_poster(r['id'], r.get('poster_path')),
            'rating': "{:.1f}".format(float(r.get('vote_average', 0)))
        }
        for r in rec_data.get('results', [])[:6]
    ]

    return jsonify({
        'movie': {
            'title': details.get('title'),
            'overview': details.get('overview'),
            'year': details.get('release_date', '')[:4],
            'rating': "{:.1f}".format(float(details.get('vote_average', 0))),
            'genres': [g['name'] for g in details.get('genres', [])],
            'director': next((c['name'] for c in credits_data.get('crew', []) if c.get('job') == 'Director'), 'Bilinmiyor'),
            'cast': cast,
            'poster': get_poster(tid, details.get('poster_path')),
            'backdrop': f"https://image.tmdb.org/t/p/original{details.get('backdrop_path')}" if details.get('backdrop_path') else None
        },
        'recommendations': recs
    })

@app.route('/recommend_advanced', methods=['POST'])
def recommend_advanced():
    try:
        data = request.json
        if not data or 'title' not in data:
            return jsonify({'results': [], 'error': 'Film adı gönderilmedi'})

        title_input = data.get('title')
        clean_t = title_input.split('(')[0].strip().lower()
        tid = title_to_tmdb_id.get(clean_t)

        if not tid:
            try:
                search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title_input}"
                r = requests.get(search_url, timeout=3).json()
                if r.get('results'):
                    tid = r['results'][0]['id']
            except Exception:
                pass

        if not tid:
            return jsonify({'results': [], 'message': 'Film bulunamadı'})

        ml_id = tmdb_to_movielens.get(tid)
        candidates = {}

        if ml_id:
            if ml_id in movie_id_to_matrix_idx:
                idx = movie_id_to_matrix_idx[ml_id]
                dist, ind = model_knn.kneighbors(movie_user_matrix.iloc[idx, :].values.reshape(1, -1), n_neighbors=15)
                for i in ind.flatten()[1:]:
                    mid = matrix_idx_to_movie_id[i]
                    candidates[mid] = 1.0

            if ml_id in nlp_indices:
                idx = nlp_indices[ml_id]
                sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
                for i in sim.argsort()[-25:-1]:
                    mid = movies.iloc[i]['movieId']
                    candidates[mid] = candidates.get(mid, 0) + sim[i]

        final_recs = []
        target_movie_data = movies[movies['tmdbId'] == tid]
        target_genres = set(target_movie_data['genres'].iloc[0].split('|')) if not target_movie_data.empty else set()
        is_kids = any(genre in target_genres for genre in ['Children', 'Animation'])

        for mid, raw_score in candidates.items():
            row_data = movies[movies['movieId'] == mid]
            if row_data.empty:
                continue
            row = row_data.iloc[0]
            cand_genres = set(row['genres'].split('|'))
            if is_kids and not any(g in cand_genres for g in ['Children', 'Animation']):
                continue

            g_sim = calculate_genre_score(target_genres, cand_genres)
            v_avg = float(row.get('vote_average', 0)) / 10
            current_final_score = (raw_score * WEIGHT_NLP) + (g_sim * WEIGHT_GENRE) + (v_avg * WEIGHT_RATING)

            final_recs.append({
                'title': row['title'],
                'tmdbId': int(row['tmdbId']),
                'score': round(current_final_score, 2),
                'poster': get_poster(row['tmdbId'], row.get('poster_path'))
            })

        final_recs = sorted(final_recs, key=lambda x: x['score'], reverse=True)[:10]
        return jsonify({'results': final_recs})

    except Exception as e:
        return jsonify({'results': [], 'error': str(e)})

@app.route('/analyze_profile', methods=['POST'])
def analyze_profile():
    try:
        answers = request.json.get('answers')
        if not answers or len(answers) != 8:
            return jsonify({'error': 'Eksik veya geçersiz veri.'}), 400

        user = UserProfile()
        for item in answers:
            q_id = int(item['q'])
            ans = item['ans'].upper()
            vector = SCENARIO_VECTORS[q_id][ans]
            user.apply_vector(vector)

        user.finalize_scores()
        result = expert_engine.get_closest(user)
        save_anonymous_log(answers, user.scores, result)
        distance = result.get('distance', 100)
        match_level = get_match_level(distance)
        similarity = get_similarity_percentage(distance)
        top_matches = expert_engine.get_top_matches(user, top_n=30)
        film_match, series_match = pick_dual_matches(top_matches)

        work_poster = None
        if result.get('source_title'):
            work_poster = get_work_match_from_title(result['source_title'])

        return jsonify({
            'archetype': result,
            'scores': user.scores,
            'normalized': [round(v, 2) for v in user.get_normalized()],
            'closest_character': result.get('closest_character'),
            'source_title': result.get('source_title'),
            'distance': result.get('distance'),
            'match_level': match_level,
            'similarity': similarity,
            'work_poster': work_poster,
            'film_match': film_match,
            'series_match': series_match,
            'dynamic_comment': generate_dynamic_comment(user.scores)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logs')
def view_logs():
    try:
        import pandas as pd

        df = pd.read_csv("logs/anon_results.csv")

        # Zaman formatını Türkiye Saati'ne (+3) çevir ve en son çözüleni en üste al
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']) + pd.Timedelta(hours=3)
            df['timestamp'] = df['timestamp'].dt.strftime('%d.%m.%Y %H:%M:%S')
            df = df.sort_values(by='timestamp', ascending=False)

        # 1. En çok çıkan arketip
        top_archetype = df['archetype'].value_counts().idxmax()

        # 2. En popüler karakter
        top_character = df['character'].value_counts().idxmax()

        # 3. Ortalama skorlar
        avg_scores = df[['uyum','ahlak','varolus','karar']].mean().round(1)

        # 4. Dağılım
        archetype_counts = df['archetype'].value_counts()

        return f"""
        <html>
        <head>
        <style>
        body {{ font-family: Arial; padding: 20px; background: #111; color: white; }}
        table {{ border-collapse: collapse; width: 100%; margin-top:20px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #444; }}
        th {{ background: #222; }}
        tr:hover {{ background: #333; }}

        .card {{
            background:#222;
            padding:15px;
            margin-bottom:15px;
            border-radius:10px;
        }}
        </style>
        </head>
        <body>

        <h1>📊 Uzman Sistem Analiz Paneli</h1>

        <div class="card">
            <h3>🔥 En Çok Çıkan Arketip</h3>
            <p>{top_archetype}</p>
        </div>

        <div class="card">
            <h3>🎬 En Popüler Karakter</h3>
            <p>{top_character}</p>
        </div>

        <div class="card">
            <h3>📈 Ortalama İnsan</h3>
            <p>
            Uyum: {avg_scores['uyum']} |
            Ahlak: {avg_scores['ahlak']} |
            Varoluş: {avg_scores['varolus']} |
            Karar: {avg_scores['karar']}
            </p>
        </div>

        <div class="card">
            <h3>📊 Arketip Dağılımı</h3>
            {archetype_counts.to_frame().to_html()}
        </div>

        <h2>📋 Tüm Kayıtlar</h2>
        {df.to_html(index=False)}

        </body>
        </html>
        """

    except Exception as e:
        return f"Hata: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5057)