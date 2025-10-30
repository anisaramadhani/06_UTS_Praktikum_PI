Information Retrieval System (CLI-Based)

📌 Deskripsi Proyek

Proyek ini merupakan implementasi sistem Information Retrieval (IR) berbasis Command-Line Interface (CLI) yang dapat melakukan pencarian dan pemeringkatan dokumen dari berbagai dataset teks nyata. Sistem ini membaca kumpulan dataset dalam format .csv, kemudian melakukan proses preprocessing, indexing, searching, dan ranking hasil menggunakan Whoosh, CountVectorizer, dan Cosine Similarity. Sistem dirancang untuk dapat mengintegrasikan konsep Vector Space Model (VSM) dengan pendekatan term weighting dan similarity computation, sehingga dapat menampilkan hasil pencarian yang relevan dan terurut berdasarkan kemiripan antara query pengguna dan isi dokumen.

🧩 Fitur Utama

1. Preprocessing Teks : Melakukan case folding, tokenization, dan stopword removal (menggunakan NLTK bahasa Indonesia). Termasuk penanganan stopword khusus agar kata penting seperti “presiden”, “kasus”, “mahasiswa” tidak dihapus.
2. Indexing Dokumen (Whoosh) : Membangun struktur indeks dari seluruh file CSV di dalam folder dataset/ agar pencarian menjadi lebih cepat dan efisien.
3. Pencarian Query (Whoosh + Cosine Similarity) : Sistem menerima query pengguna, memprosesnya, lalu mencocokkannya dengan dokumen di indeks. Hasil kemudian diranking berdasarkan nilai cosine similarity antara query dan dokumen.
4. Fallback Pencarian Manual : Jika Whoosh tidak menemukan hasil, sistem melakukan pencarian substring langsung pada isi CSV agar tidak ada query yang terlewat.
5. Tabel Hasil Ranking : Menampilkan hasil 5 dokumen teratas beserta skor kemiripan dalam format tabel di terminal.

🏗️ Arsitektur Sistem

Alur kerja sistem adalah sebagai berikut:
Pengguna → Preprocessing Query → Pencarian pada Whoosh Index → Pembobotan TF (CountVectorizer) → Perhitungan Cosine Similarity → Ranking Dokumen → Output di Terminal

Komponen Utama:
1. Preprocessing Layer : Membersihkan teks dan menghapus stopwords.
2. Indexing Layer : Mengonversi dataset ke struktur indeks Whoosh.
3. Retrieval Layer : Melakukan pencarian, perhitungan kemiripan, dan penentuan ranking.
4. CLI Interface : Menyediakan antarmuka sederhana di terminal dengan menu interaktif.

🚀 Cara Menjalankan Program

Jalankan program melalui terminal / command prompt:
python ir_system.py

Kemudian pilih menu yang tersedia:

*=== INFORMATION RETRIEVAL SYSTEM ===
[1] Load & Index Dataset
[2] Search Query
[3] Exit
====================================*

🧾 Contoh Penggunaan
1️⃣ Membuat Index:

Pilih menu (1/2/3): 1
[INFO] Indexed kompas.csv (kolom: konten, 51234 baris)
[INFO] Indexed tempo.csv (kolom: isi, 48000 baris)
[INFO] Indexing selesai. Total dokumen terindeks: 99234

2️⃣ Mencari Query:

Pilih menu (1/2/3): 2
Masukkan query pencarian: pendidikan

=== Top 5 Hasil Pencarian (CountVectorizer + Cosine) ===
1. kompas.csv_row_2312 (Skor: 0.3721)
2. etd_ugm.csv_row_12451 (Skor: 0.3615)
3. mojok.csv_row_115 (Skor: 0.2958)
4. tempo.csv_row_892 (Skor: 0.2871)
5. etd_usk.csv_row_9543 (Skor: 0.2749)

📦 File Penting

ir_system.py : File utama sistem IR berbasis CLI
requirements.txt : Daftar pustaka Python yang digunakan
dataset/ : Folder berisi dataset .csv
indexdir/ : Folder hasil indexing otomatis oleh Whoosh

🧰 Dependensi Utama

Berikut pustaka yang digunakan:
- pandas – membaca dataset .csv
- whoosh – indexing dan searching dokumen
- nltk – preprocessing teks (stopwords, tokenisasi)
- scikit-learn – CountVectorizer dan Cosine Similarity

Contoh requirements.txt:
pandas>=2.0.0
whoosh>=2.7.4
nltk>=3.8.1
scikit-learn>=1.3.0

🧪 Pengujian Query

Sistem diuji menggunakan lima dataset:
1. etd_ugm.csv
2. etd_usk.csv
3. kompas.csv
4. tempo.csv
5. mojok.csv

Hasil pengujian menunjukkan bahwa sistem mampu memproses puluhan ribu dokumen dan menghasilkan pencarian dengan waktu respon cepat (<3 detik untuk query tunggal) serta hasil relevan sesuai konteks kata kunci.

Kesimpulan

Sistem Information Retrieval berbasis CLI ini berhasil mengimplementasikan konsep Vector Space Model dengan perhitungan Cosine Similarity secara efektif. Sistem dapat membaca berbagai dataset .csv, melakukan preprocessing teks Bahasa Indonesia, membangun indeks dengan Whoosh, dan memberikan hasil pencarian relevan dengan efisiensi tinggi.
