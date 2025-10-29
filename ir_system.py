# ir_system.py
import os
import string
import re
import pandas as pd
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Download stopwords jika belum ada (skip jika error koneksi)
try:
    nltk.download("stopwords", quiet=True)
except:
    pass  # Abaikan error download, asumsi stopwords sudah terinstall

# =============== PREPROCESSING ===============
def preprocess(text):
    """Case folding, tokenization, stopword removal"""
    if not isinstance(text, str):
        text = str(text)
    # Be robust to malformed spacing like 'pesawatUnited' -> insert a space
    # between a lowercase letter followed by an uppercase letter
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # normalize whitespace and lower-case
    text = " ".join(text.split())
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Custom stopwords: hapus kata-kata penting dari stopwords default
    stop_words = set(stopwords.words("indonesian"))
    # Kata-kata penting yang tidak boleh dihapus
    important_words = {
        # Hukum & Kriminal
        'kasus', 'polisi', 'jaksa', 'hakim', 'pengadilan', 'tersangka', 
        'terdakwa', 'korban', 'saksi', 'bukti', 'hukum', 'pidana', 'perdata',
        'vonis', 'tuntutan', 'pengacara', 'kuasa', 'pasal',
        # Politik & Pemerintahan
        'presiden', 'menteri', 'gubernur', 'bupati', 'walikota', 'dpr', 'dprd',
        'pemerintah', 'negara', 'rakyat', 'pemilu', 'pilkada', 'partai',
        # Ekonomi & Bisnis
        'ekonomi', 'bisnis', 'pasar', 'harga', 'rupiah', 'inflasi', 'bank',
        'investasi', 'saham', 'perusahaan', 'industri', 'perdagangan',
        # Kesehatan
        'pasien', 'rumah', 'sakit', 'dokter', 'obat', 'virus', 'covid',
        'vaksin', 'pandemi', 'wabah', 'kesehatan',
        # Pendidikan
        'mahasiswa', 'siswa', 'guru', 'dosen', 'universitas', 'sekolah',
        'kampus', 'pendidikan', 'kuliah', 'ujian',
        # Umum
        'data', 'informasi', 'media', 'berita', 'laporan', 'hasil', 'proses',
        'sistem', 'program', 'kegiatan', 'masalah', 'solusi', 'dampak'
    }
    stop_words = stop_words - important_words
    
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# =============== DETEKSI KOLUMN TEKS ===============
def detect_text_column(df):
    """Deteksi kolom teks utama otomatis"""
    if df.shape[1] == 1:
        return df.columns[0]
    max_len, text_col = 0, None
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].astype(str).str.len().mean()
            if avg_len > max_len:
                max_len = avg_len
                text_col = col
    return text_col

# =============== INDEXING WHOOSH ===============
def create_index(dataset_dir="dataset", index_dir="indexdir"):
    """Membuat index Whoosh dari semua CSV di folder dataset"""
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    else:
        # hapus isi lama supaya tidak tumpang tindih
        for f in os.listdir(index_dir):
            os.remove(os.path.join(index_dir, f))

    schema = Schema(title=ID(stored=True), path=ID(stored=True), content=TEXT(stored=True))
    ix = index.create_in(index_dir, schema)
    writer = ix.writer()

    total_docs = 0
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv"):
                path = os.path.join(root, file)
                try:
                    # baca CSV aman untuk teks panjang
                    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", quotechar='"')
                    col = detect_text_column(df)
                    if not col:
                        print(f"[WARNING] Kolom teks tidak ditemukan di {file}")
                        continue
                    for i, row in df.iterrows():
                        content = preprocess(str(row[col]))
                        if content.strip() == "":
                            continue
                        title = f"{file}_row_{i}"
                        writer.add_document(title=title, path=path, content=content)
                        total_docs += 1
                    print(f"[INFO] Indexed {file} (kolom: {col}, {len(df)} baris)")
                except Exception as e:
                    print(f"[ERROR] {file}: {e}")

    writer.commit()
    print(f"\n[INFO] Indexing selesai. Total dokumen terindeks: {total_docs}")
    print(f"[INFO] File index tersimpan dalam folder: '{index_dir}'\n")

# =============== SEARCH & RANKING ===============
def search_query(index_dir, query):
    """Cari & ranking hasil menggunakan Whoosh + CountVectorizer + Cosine Similarity"""
    if not os.path.exists(index_dir):
        print(f"[ERROR] Direktori index '{index_dir}' tidak ditemukan. Jalankan menu indexing terlebih dahulu.")
        return

    ix = index.open_dir(index_dir)
    qp = QueryParser("content", schema=ix.schema)
    q = qp.parse(preprocess(query))

    with ix.searcher() as searcher:
        results = searcher.search(q, limit=None)
        docs = [(r["title"], r["content"]) for r in results]

        if not docs:
            # Fallback: jika Whoosh tidak menemukan dokumen (mis-tokenization karena
            # data CSV rusak / tanpa spasi), lakukan pencarian substring langsung
            # pada file-file CSV di folder 'dataset'. Ini menangkap kasus seperti
            # 'pesawatUnited' atau 'Airlinesmendarat' yang ter-join tanpa spasi.
            print("[INFO] Tidak ada dokumen ditemukan oleh Whoosh. Mencoba fallback substring scan pada CSV...")
            found = []
            terms = [t.lower() for t in query.split() if t.strip()]
            dataset_dir = "dataset"
            for root, _, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.csv'):
                        path = os.path.join(root, file)
                        try:
                            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip', quotechar='"')
                            col = detect_text_column(df)
                            if not col:
                                continue
                            for i, row in df.iterrows():
                                raw = str(row[col]).lower()
                                if all(t in raw for t in terms):
                                    title = f"{file}_row_{i}"
                                    found.append((title, preprocess(raw)))
                        except Exception:
                            continue

            if not found:
                print("[INFO] Tidak ada dokumen ditemukan untuk query tersebut.")
                return

            docs = found

        print(f"[INFO] Ditemukan {len(docs)} dokumen relevan untuk query '{query}'.")

        # Representasi dokumen (CountVectorizer)
        corpus = [d[1] for d in docs]
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(corpus + [preprocess(query)])

        # Hitung cosine similarity
        cosine_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

        # Urutkan berdasarkan skor tertinggi
        ranked = sorted(zip(docs, cosine_scores), key=lambda x: x[1], reverse=True)[:5]

        print("\n=== Top 5 Hasil Pencarian (CountVectorizer + Cosine) ===")
        for i, ((title, _), score) in enumerate(ranked, 1):
            print(f"{i}. {title} (Skor: {score:.4f})")
        print()

# =============== CLI INTERFACE ===============
def main():
    dataset_dir = "dataset"  # folder kamu sekarang bernama 'dataset'
    index_dir = "indexdir"

    while True:
        print("\n=== INFORMATION RETRIEVAL SYSTEM ===")
        print("[1] Load & Index Dataset")
        print("[2] Search Query")
        print("[3] Exit")
        print("====================================")
        choice = input("Pilih menu (1/2/3): ").strip()

        if choice == "1":
            create_index(dataset_dir, index_dir)
        elif choice == "2":
            query = input("Masukkan query pencarian: ")
            search_query(index_dir, query)
        elif choice == "3":
            print("Keluar dari sistem.")
            break
        else:
            print("Pilihan tidak valid, coba lagi.")

if __name__ == "__main__":
    main()