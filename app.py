# file: app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import pytz
import joblib
import os
import re
import string
import requests
from bs4 import BeautifulSoup
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Klasifikasi Berita", page_icon="📰", layout="wide")

# --- PATHS & TIMEZONE (LOKASI YANG BENAR) ---
ORIGINAL_DATA_PATH = 'data_sintetis_topic_news_kontan.csv'
PREDICTED_DATA_PATH = 'predicted_articles.csv'
WIB = pytz.timezone('Asia/Jakarta')

# --- FUNGSI-FUNGSI HELPER (Backend) ---

@st.cache_data(ttl=3600) # Cache selama 1 jam
def load_custom_stopwords(url):
    """Mengunduh dan memproses daftar stopwords kustom dari URL."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            stopwords = set(response.text.splitlines())
            return stopwords
    except requests.RequestException:
        return set()
    return set()

def initialize_predicted_csv():
    if not os.path.exists(PREDICTED_DATA_PATH):
        df = pd.DataFrame(columns=['judul', 'url', 'content', 'tag', 'datetime', 'article', 'label_topic'])
        df.to_csv(PREDICTED_DATA_PATH, index=False, sep=';')

@st.cache_data(ttl=30)
def load_combined_data():
    try:
        df_original = pd.read_csv(ORIGINAL_DATA_PATH, delimiter=';')
        initialize_predicted_csv()
        if os.path.exists(PREDICTED_DATA_PATH) and os.path.getsize(PREDICTED_DATA_PATH) > 0:
            try:
                df_predicted = pd.read_csv(PREDICTED_DATA_PATH, delimiter=';')
                if not df_predicted.empty:
                    df_combined = pd.concat([df_original, df_predicted], ignore_index=True)
                else:
                    df_combined = df_original
            except pd.errors.EmptyDataError:
                df_combined = df_original
        else:
            df_combined = df_original
        
        df_combined['datetime'] = pd.to_datetime(df_combined['datetime'], errors='coerce')
        df_combined.dropna(subset=['datetime'], inplace=True)
        df_combined['date'] = df_combined['datetime'].dt.date
        return df_combined
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: `{e.filename}`.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Terjadi error tak terduga saat memuat data: {e}")
        return pd.DataFrame()

def save_prediction(article_text, prediction, url="-", title="Teks Manual"):
    now_wib = datetime.now(WIB)
    new_data = {
        'judul': [title], 'url': [url], 'content': [article_text], 'tag': [prediction],
        'datetime': [now_wib.strftime('%Y-%m-%d %H:%M:%S')], 'article': [article_text], 'label_topic': [prediction]
    }
    new_df = pd.DataFrame(new_data)
    header = not os.path.exists(PREDICTED_DATA_PATH) or os.path.getsize(PREDICTED_DATA_PATH) == 0
    new_df.to_csv(PREDICTED_DATA_PATH, mode='a', header=header, index=False, sep=';')
    st.cache_data.clear()

@st.cache_resource
def load_classification_model_and_stopwords():
    try:
        model_path = 'model_svc/linear_svc_model.joblib'
        vectorizer_path = 'model_svc/tfidf_vectorizer.joblib'
        ml_model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        factory = StopWordRemoverFactory()
        sastrawi_stopwords = factory.get_stop_words()
        
        custom_stopwords_url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-id/master/stopwords-id.txt"
        custom_stopwords = load_custom_stopwords(custom_stopwords_url)
        
        manual_stopwords = {
            # Abjad dan angka
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            # Kata umum yang sering muncul di berita
            'dan', 'atau', 'dengan', 'untuk', 'pada',   'di', 'yang', 'adalah', 'ini', 'itu',
            'sebagai', 'juga', 'tersebut', 'dari', 'ke', 'oleh', 'dalam', 'mengenai', 'tentang',
            'seperti', 'hingga', 'selain', 'antara', 'terhadap', 'dengan', 'mengenai',
            # Kata umum terkait berita
            'berita', 'artikel', 'kontan', 'berita kontan', 'kontan.co.id', 'berita kontan.co.id',
            'berita terkini', 'berita terbaru', 'berita hari ini', 'berita ekonomi',
            'berita bisnis', 'berita politik', 'berita olahraga', 'berita teknologi', 'berita kesehatan',
            'berita hiburan', 'berita nasional', 'berita internasional', 'berita daerah',
            'berita dunia', 'berita terbaru hari ini', 'berita populer', 'berita viral',
            'berita hangat', 'berita menarik', 'berita eksklusif',
            'berita terbaru kontan', 'berita kontan hari ini', 'berita kontan terbaru',
            'berita kontan.co.id', 'kontan.co.id berita', 'kontan.co.id terbaru',
            'kontan.co.id hari ini', 'kontan.co.id ekonomi', 'kontan.co.id bisnis',
            'kontan.co.id politik', 'kontan.co.id olahraga', 'kontan.co.id teknologi',
            'kontan.co.id kesehatan', 'kontan.co.id hiburan', 'kontan.co.id nasional',
            'kontan.co.id internasional', 'kontan.co.id daerah', 'kontan.co.id dunia',
            'kontan.co.id populer', 'kontan.co.id viral', 'kontan.co.id hangat',
            'kontan.co.id menarik', 'kontan.co.id eksklusif',
            # Kata umum terkait ekonomi
            'ekonomi', 'bisnis', 'investasi', 'saham', 'pasar', 'industri', 'perdagangan',
            'keuangan', 'bank', 'perbankan', 'uang', 'modal',   'laba', 'rugi', 'pertumbuhan',
            'inflasi', 'deflasi',
            'produk', 'jasa', 'kredit', 'utang', 'piutang', 'aset', 'liabilitas', 'ekuitas',
            'revenue', 'pendapatan', 'pengeluaran', 'biaya', 'profit', 'margin', 'omzet',
            'penjualan', 'pemasaran', 'strategi', 'analisis', 'laporan', 'data', 'statistik',
            'riset', 'tren', 'proyeksi', 'forecast', 'kinerja', 'evaluasi', 'audit',
            'regulasi', 'peraturan', 'pajak', 'subsidi', 'insentif', 'kebijakan', 'reformasi',
            'pertumbuhan ekonomi', 'kebijakan ekonomi', 'indeks harga', 'kurs', 'nilai tukar',
            'suku bunga', 'cadangan devisa', 'neraca perdagangan', 'defisit anggaran',
            'surplus anggaran', 'utang luar negeri', 'investasi asing', 'ekspor', 'impor',
            'kinerja perusahaan', 'laporan keuangan', 'analisis pasar', 'riset pasar',
            'strategi bisnis', 'analisis industri', 'laporan tahunan', 'laporan kuartalan',
            'laporan bulanan', 'laporan triwulanan', 'laporan semesteran', 'laporan tahunan',
            'laporan keuangan tahunan', 'laporan keuangan kuartalan', 'laporan keuangan bulanan',
            'laporan keuangan triwulanan', 'laporan keuangan semesteran', 'laporan keuangan tahunan',
            'laporan keuangan perusahaan', 'laporan keuangan bank', 'laporan keuangan industri',
            'laporan keuangan sektor', 'laporan keuangan ekonomi', 'laporan keuangan bisnis',
            'laporan keuangan investasi', 'laporan keuangan pasar', 'laporan keuangan regulasi',
            'laporan keuangan peraturan', 'laporan keuangan pajak', 'laporan keuangan subsidi',
            'laporan keuangan insentif', 'laporan keuangan kebijakan', 'laporan keuangan reformasi',
            'laporan keuangan pertumbuhan ekonomi', 'laporan keuangan kebijakan ekonomi',
            'laporan keuangan indeks harga', 'laporan keuangan kurs', 'laporan keuangan nilai tukar',
            'laporan keuangan suku bunga', 'laporan keuangan cadangan devisa',
            'laporan keuangan neraca perdagangan', 'laporan keuangan defisit anggaran',
            'laporan keuangan surplus anggaran', 'laporan keuangan utang luar negeri',
            'laporan keuangan investasi asing', 'laporan keuangan ekspor', 'laporan keuangan impor',
            'laporan keuangan kinerja perusahaan', 'laporan keuangan analisis pasar',
            'laporan keuangan riset pasar', 'laporan keuangan strategi bisnis',
            'laporan keuangan analisis industri', 'laporan keuangan laporan tahunan',
            'laporan keuangan laporan kuartalan', 'laporan keuangan laporan bulanan',
            'laporan keuangan laporan triwulanan', 'laporan keuangan laporan semesteran',
            'laporan keuangan laporan tahunan', 'laporan keuangan laporan keuangan tahunan',
            'laporan keuangan laporan keuangan kuartalan', 'laporan keuangan laporan keuangan bulanan',
            'laporan keuangan laporan keuangan triwulanan', 'laporan keuangan laporan keuangan semesteran',
            'laporan keuangan laporan keuangan tahunan', 'laporan keuangan laporan keuangan perusahaan',
            'laporan keuangan laporan keuangan bank', 'laporan keuangan laporan keuangan industri',
            'laporan keuangan laporan keuangan sektor', 'laporan keuangan laporan keuangan ekonomi',
            'laporan keuangan laporan keuangan bisnis', 'laporan keuangan laporan keuangan investasi',
            'laporan keuangan laporan keuangan pasar', 'laporan keuangan laporan keuangan regulasi',
            'laporan keuangan laporan keuangan peraturan', 'laporan keuangan laporan keuangan pajak',
            'laporan keuangan laporan keuangan subsidi', 'laporan keuangan laporan keuangan insentif',
            'laporan keuangan laporan keuangan kebijakan', 'laporan keuangan laporan keuangan reformasi',
            'laporan keuangan laporan keuangan pertumbuhan ekonomi', 'laporan keuangan laporan keuangan kebijakan ekonomi',
            'laporan keuangan laporan keuangan indeks harga', 'laporan keuangan laporan keuangan kurs',
            'laporan keuangan laporan keuangan nilai tukar', 'laporan keuangan laporan keuangan suku bunga',
            'laporan keuangan laporan keuangan cadangan devisa', 'laporan keuangan laporan keuangan neraca perdagangan',
            'laporan keuangan laporan keuangan defisit anggaran', 'laporan keuangan laporan keuangan surplus anggaran',
            'laporan keuangan laporan keuangan utang luar negeri', 'laporan keuangan laporan keuangan investasi asing',
            'laporan keuangan laporan keuangan ekspor', 'laporan keuangan laporan keuangan impor',
            'laporan keuangan laporan keuangan kinerja perusahaan', 'laporan keuangan laporan keuangan analisis pasar',
            'laporan keuangan laporan keuangan riset pasar', 'laporan keuangan laporan keuangan strategi bisnis',
            'laporan keuangan laporan keuangan analisis industri', 'laporan keuangan laporan keuangan laporan tahunan',
            'laporan keuangan laporan keuangan laporan kuartalan', 'laporan keuangan laporan keuangan laporan bulanan',
            'laporan keuangan laporan keuangan laporan triwulanan', 'laporan keuangan laporan keuangan laporan semesteran',
            'laporan keuangan laporan keuangan laporan tahunan', 'laporan keuangan laporan keuangan laporan keuangan tahunan',
            'laporan keuangan laporan keuangan laporan keuangan kuartalan', 'laporan keuangan laporan keuangan laporan keuangan bulanan',
            'laporan keuangan laporan keuangan laporan keuangan triwulanan', 'laporan keuangan laporan keuangan laporan keuangan semesteran',
            'laporan keuangan laporan keuangan laporan keuangan tahunan', 'laporan keuangan laporan keuangan laporan keuangan perusahaan',
            'laporan keuangan laporan keuangan laporan keuangan bank', 'laporan keuangan laporan keuangan laporan keuangan industri',
            'laporan keuangan laporan keuangan laporan keuangan sektor', 'laporan keuangan laporan keuangan laporan keuangan ekonomi',
            'laporan keuangan laporan keuangan laporan keuangan bisnis', 'laporan keuangan laporan keuangan laporan keuangan investasi',
            'laporan keuangan laporan keuangan laporan keuangan pasar', 'laporan keuangan laporan keuangan laporan keuangan regulasi',
            'laporan keuangan laporan keuangan laporan keuangan peraturan', 'laporan keuangan laporan keuangan laporan keuangan pajak',
            'laporan keuangan laporan keuangan laporan keuangan subsidi', 'laporan keuangan laporan keuangan laporan keuangan insentif',
            'laporan keuangan laporan keuangan laporan keuangan kebijakan', 'laporan keuangan laporan keuangan laporan keuangan reformasi',
            'laporan keuangan laporan keuangan laporan keuangan pertumbuhan ekonomi',
            'laporan keuangan laporan keuangan laporan keuangan kebijakan ekonomi',
            'laporan keuangan laporan keuangan laporan keuangan indeks harga',
            'laporan keuangan laporan keuangan laporan keuangan kurs', 'laporan keuangan laporan keuangan laporan keuangan nilai tukar',
            'laporan keuangan laporan keuangan laporan keuangan suku bunga',
            'laporan keuangan laporan keuangan laporan keuangan cadangan devisa',
            'laporan keuangan laporan keuangan laporan keuangan neraca perdagangan',
            # Kata umum terkait media/berita
            'kontan', 'co', 'id', 'jakarta', 'baca', 'juga', 'lainnya', 'artikel', 
            'news', 'editor', 'penulis', 'lihat', 'hal', 'sumber', 'cnn', 'indonesia',
            'detik', 'kompas', 'liputan', 'tribun', 'bisnis', 'ekonomi', 'rp', 'usd',
            'dikutip', 'dilansir', 'senilai', 'sebesar', 'yakni', 'yaitu', 'berita',
            'cek', 'google', 'trans', 'tv', 'motion', 'grafik',
            'berita kontan', 'kontan.co.id', 'kontan.co.id berita', 'kontan.co.id terbaru',
            'kontan.co.id hari ini', 'kontan.co.id ekonomi', 'kontan.co.id bisnis',
            'kontan.co.id politik', 'kontan.co.id olahraga', 'kontan.co.id teknologi',
            'kontan.co.id kesehatan', 'kontan.co.id hiburan', 'kontan.co.id nasional',
            
            # Kata kerja peliputan & Jabatan
            'mengatakan', 'menurut', 'ujar', 'kata', 'jelasnya', 'katanya', 'menambahkan',
            'ungkapnya', 'tuturnya', 'presiden', 'direktur', 'utama', 'menteri', 'kepala',
            'gubernur', 'sekretaris', 'jenderal',
            'kepala daerah', 'anggota', 'dewan', 'komisi', 'anggota dewan', 'anggota komisi',
            'anggota legislatif', 'anggota parlemen', 'anggota DPR', 'anggota DPD',
            'anggota MPR', 'anggota DPRD', 'anggota DPD RI', 'anggota MPR RI', 'anggota DPR RI',
            'anggota legislatif daerah', 'anggota parlemen daerah', 'anggota dewan daerah',
            'anggota komisi daerah', 'anggota dewan perwakilan daerah', 'anggota komisi perwakilan daerah',
            'anggota dewan perwakilan rakyat', 'anggota komisi perwakilan rakyat',
            'anggota dewan perwakilan rakyat daerah', 'anggota komisi perwakilan rakyat daerah',
            'anggota dewan perwakilan rakyat daerah provinsi', 'anggota komisi perwakilan rakyat daerah provinsi',
            
            # Satuan waktu & Nama Bulan
            'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu',
            'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 
            'agustus', 'september', 'oktober', 'november', 'desember',
            'tahun', 'bulan', 'pekan', 'hari', 'kemarin', 'lalu', 'mendatang', 'sebelumnya',
            'minggu ini', 'bulan ini', 'tahun ini', 'minggu lalu', 'bulan lalu', 'tahun lalu',
            'minggu depan', 'bulan depan', 'tahun depan', 'hari ini', 'hari kemarin',
            'hari ini', 'hari mendatang', 'hari sebelumnya', 'hari lalu', 'hari depan',
            
            # Satuan jumlah
            'juta', 'miliar', 'triliun', 'persen', 'kali', 'persero', 'tbk',
            'rupiah', 'dolar', 'dolar amerika', 'dolar as', 'dolar indonesia', 'dolar singapura',
            'dolar australia', 'dolar kanada', 'dolar hongkong', 'dolar eropa',
            'dolar inggris', 'dolar jepang', 'dolar cina', 'dolar malaysia', 'dolar filipina',
            'dolar thailand', 'dolar vietnam', 'dolar brunei', 'dolar selandia baru',
            'dolar swiss', 'dolar arab saudi', 'dolar uae', 'dolar qatar', 'dolar kuwait',
            'dolar bahrain', 'dolar oman', 'dolar yaman', 'dolar mesir', 'dolar libanon',
            'dolar turki', 'dolar yunani', 'dolar italia',
            'dolar spanyol', 'dolar portugal', 'dolar belanda', 'dolar jerman',
            'dolar prancis', 'dolar inggris', 'dolar irlandia', 'dolar skotlandia',
            'dolar wales', 'dolar islandia', 'dolar finlandia', 'dolar norwegia',
            'dolar denmark', 'dolar swedia', 'dolar polandia', 'dolar ceko',
            'dolar hungaria', 'dolar rumania', 'dolar bulgaria', 'dolar kroasia',
            
            # Kata umum lainnya yang sering muncul
            'antara', 'sementara', 'sehingga', 'meski', 'namun', 'tersebut', 'kepada',
            'saat', 'akan', 'masih', 'bisa', 'telah', 'hingga', 'jadi', 'bakal',
            'kini', 'saja', 'pun', 'sangat', 'lebih', 'kurang', 'cukup', 'perlu',
            'membuat', 'menjadi', 'merupakan', 'adalah', 'yakni', 'yaitu'
        }
        
        combined_stopwords = set(sastrawi_stopwords).union(custom_stopwords).union(manual_stopwords)
        
        return ml_model, vectorizer, factory.create_stop_word_remover(), combined_stopwords
    except Exception as e:
        st.error(f"Error saat memuat model atau stopwords: {e}.")
        return None, None, None, set()

def preprocess_text(text, stopword_remover):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text); text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text); text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text); text = re.sub(r'\w*\d\w*', '', text)
    return str(stopword_remover.remove(text))

def predict_topic_and_proba(prediction_input_text, _ml_model, _vectorizer, _stopword_remover):
    cleaned_text = preprocess_text(prediction_input_text, _stopword_remover)
    text_vector = _vectorizer.transform([cleaned_text])
    
    prediction = _ml_model.predict(text_vector)[0]
    probabilities = _ml_model.predict_proba(text_vector)[0]
    
    proba_df = pd.DataFrame({
        'Topik': _ml_model.classes_, 'Probabilitas': probabilities
    }).sort_values(by='Probabilitas', ascending=False).reset_index(drop=True)
    proba_df['Probabilitas'] = proba_df['Probabilitas'].apply(lambda x: f"{x:.2%}")
    return prediction, proba_df

def display_prediction_result(original_article_text, ml_model, vectorizer, stopword_remover, url="-", title="Teks Manual"):
    with st.spinner("Menganalisis dan memprediksi topik..."):
        try:
            # --- PERUBAHAN DI SINI ---
            # Menentukan teks yang akan disimpan. Jika input manual, simpan teks asli saja.
            # Jika dari scrape URL, simpan judul dan teksnya.
            if title == "Teks Manual":
                prediction_input = original_article_text
            else:
                prediction_input = title + ". " + original_article_text
            # --------------------------

            prediction, proba_df = predict_topic_and_proba(prediction_input, ml_model, vectorizer, stopword_remover)
            st.success(f"**Prediksi Topik Utama:** `{prediction}`")
            with st.expander("Lihat Detail Probabilitas Prediksi"):
                st.dataframe(proba_df, use_container_width=True)
            
            # Simpan teks asli (original_article_text) ke dalam file CSV
            save_prediction(original_article_text, prediction, url=url, title=title)
            st.info("Hasil prediksi telah disimpan.", icon="💾")
            
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
            st.warning("Pastikan model Anda mendukung `predict_proba`. Latih ulang model dengan `SVC(probability=True)` jika diperlukan.")

def scrape_kontan_article(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200: return None
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Coba ambil judul dari tag <title> atau <h1>
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else url.split('/')[-1].replace('-', ' ').capitalize()
        
        possible_selectors = ['.detail-desk.pr-3', 'div.tm-content', 'div.detail_desk', '#content', 'div.detail-konten']
        article_text = ''
        for selector in possible_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                paragraphs = content_div.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if len(article_text.split()) > 20: 
                    return title, article_text
        
        if not article_text:
            all_paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in all_paragraphs])

        return (title, article_text) if article_text else (None, None)

    except requests.RequestException:
        return None, None

# --- UI STREAMLIT ---
ml_model, vectorizer, stopword_remover, combined_stopwords = load_classification_model_and_stopwords()
df_main = load_combined_data()

# --- SIDEBAR SEBAGAI PUSAT FILTER ---
with st.sidebar:
    st.title("MENU & FILTER")
    page = st.radio("Pilih Halaman:", ["Dashboard", "Klasifikasi Teks"])
    st.markdown("---")

    if not df_main.empty:
        st.header("Filter Data")
        min_date, max_date = df_main['date'].min(), df_main['date'].max()
        date_range = st.date_input("Pilih Rentang Tanggal", (min_date, max_date), min_value=min_date, max_value=max_date, format="YYYY-MM-DD")
        
        all_topics = sorted(df_main['label_topic'].unique())
        
        if 'select_all' not in st.session_state: st.session_state.select_all = True

        if st.checkbox("Pilih Semua Topik", key='select_all'):
            selected_topics = st.multiselect("Pilih Topik", all_topics, default=all_topics)
        else:
            selected_topics = st.multiselect("Pilih Topik", all_topics)
    else:
        st.warning("Data belum tersedia untuk difilter.")

# Filter data berdasarkan input sidebar
if not df_main.empty and 'date_range' in locals() and len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_main[
        (df_main['date'] >= start_date) &
        (df_main['date'] <= end_date) &
        (df_main['label_topic'].isin(selected_topics))
    ].copy()
else:
    df_filtered = df_main.copy()

# --- HALAMAN DASHBOARD ---
if page == "Dashboard":
    st.title("DASHBOARD KLASIFIKASI TOPIK BERITA KONTAN")
    st.header("Analisis Umum")
    if not df_filtered.empty:
        col1, col2 = st.columns(2)
        col1.metric("Total Artikel", f"{len(df_filtered)} 📄")
        col2.metric("Total Topik", f"{df_filtered['label_topic'].nunique()} 🏷️")
        st.markdown("<hr>", unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns([1, 1.2])
        with viz_col1:
            st.subheader("Distribusi Topik")
            topic_counts = df_filtered['label_topic'].value_counts()
            fig_pie = px.pie(topic_counts, names=topic_counts.index, values=topic_counts.values, hole=.4,
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='outside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=True, legend_title_text='Topik Berita',
                                  margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with viz_col2:
            st.subheader(f"Word Cloud Topik")
            with st.spinner("Membuat Word Cloud..."):
                text = ' '.join(df_filtered['article'].dropna())
                if text:
                    try:
                        wordcloud = WordCloud(width=800, height=500, background_color='white', colormap='viridis', stopwords=combined_stopwords).generate(text)
                        fig_wc, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig_wc)
                    except Exception as e:
                        st.warning(f"Tidak dapat membuat word cloud: {e}")
                else:
                    st.info("Tidak ada teks pada data untuk membuat Word Cloud.")
        
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Detail Data Artikel")
        
        df_display = df_filtered.copy()
        
        # --- PERUBAHAN DI SINI ---
        # Membuat kolom 'artikel' hanya jika 'judul' ada, jika tidak gunakan 'content'
        if 'judul' in df_display.columns:
             df_display['artikel'] = df_display['judul'].fillna('') + ". " + df_display['content'].fillna('')
        else:
             df_display['artikel'] = df_display['content'].fillna('')
             
        df_display['datetime'] = pd.to_datetime(df_display['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_display[['datetime', 'artikel', 'label_topic', 'url']], use_container_width=True)
    else:
        st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih.")

# --- HALAMAN KLASIFIKASI ---
elif page == "Klasifikasi Teks":
    st.title("KLASIFIKASI TOPIK BERITA")
    if ml_model:
        input_manual, input_url, input_csv = st.tabs(["✍️ Input Manual", "🔗 Scrape dari URL", "📂 Unggah File CSV"])
        with input_manual:
            st.subheader("Input Teks Manual")
            if 'manual_text' not in st.session_state: st.session_state.manual_text = ""
            def clear_text(): st.session_state.manual_text = ""
            st.text_area("Masukkan Teks Artikel di Bawah Ini:", height=250, key="manual_text")
            btn_col1, btn_col2 = st.columns([1, 0.3])
            with btn_col1:
                if st.button("Prediksi Teks", type="primary", use_container_width=True):
                    if st.session_state.manual_text:
                        display_prediction_result(st.session_state.manual_text, ml_model, vectorizer, stopword_remover)
                    else:
                        st.warning("Harap masukkan teks artikel terlebih dahulu.")
            with btn_col2: st.button("Hapus Teks", on_click=clear_text, use_container_width=True)

        with input_url:
            st.subheader("Scrape Artikel dari URL")
            url_input = st.text_input("Masukkan URL Artikel dari Kontan.co.id:")
            if st.button("Scrape & Prediksi", type="primary"):
                if url_input:
                    title_scrape, article_text_scrape = scrape_kontan_article(url_input)
                    if article_text_scrape:
                        st.success("Berhasil mengambil teks artikel!")
                        with st.expander("Lihat Teks yang Di-scrape"): st.write(article_text_scrape)
                        display_prediction_result(article_text_scrape, ml_model, vectorizer, stopword_remover, url=url_input, title=title_scrape)
                    else:
                        st.error("Gagal mengambil teks dari URL. Pastikan URL valid atau struktur website tidak berubah drastis.")
                else:
                    st.warning("Harap masukkan URL terlebih dahulu.")
        
        with input_csv:
            st.subheader("Prediksi dari File CSV")
            st.info("Pastikan file CSV Anda memiliki kolom 'judul' dan 'content' (atau 'article').")
            uploaded_file = st.file_uploader("Pilih file CSV:", type="csv")
            if uploaded_file:
                try:
                    df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
                    text_column = 'article' if 'article' in df_upload.columns else 'content'
                    if text_column not in df_upload.columns:
                        st.error("Gagal menemukan kolom 'article' atau 'content' di dalam file CSV.")
                    else:
                        st.write("Pratinjau Data Unggahan:", df_upload.head())
                        if st.button("Mulai Prediksi", type="primary"):
                            predictions = []
                            progress_bar = st.progress(0, text="Memproses prediksi...")
                            total_rows = len(df_upload)
                            for i, row in df_upload.iterrows():
                                original_content = row[text_column]
                                title_csv = row.get('judul', f"Artikel Baris {i+1}")
                                if pd.notna(original_content):
                                    prediction_input_csv = str(title_csv) + ". " + str(original_content)
                                    prediction, _ = predict_topic_and_proba(prediction_input_csv, ml_model, vectorizer, stopword_remover)
                                    save_prediction(original_content, prediction, url=row.get('url', '-'), title=title_csv)
                                else:
                                    prediction = None
                                predictions.append(prediction)
                                progress_bar.progress((i + 1) / total_rows, text=f"Memproses baris {i+1}/{total_rows}")
                            df_upload['predicted_topic'] = predictions
                            st.success("Prediksi selesai! Semua hasil telah disimpan.")
                            st.dataframe(df_upload)
                            st.download_button("Unduh Hasil Prediksi (CSV)", df_upload.to_csv(index=False).encode('utf-8'), 'hasil_prediksi.csv', 'text/csv')
                except Exception as e:
                    st.error(f"Terjadi error saat memproses file: {e}")
    else:
        st.error("Model klasifikasi tidak dapat dimuat.")