import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import joblib
import os
import re
import string
import requests
from bs4 import BeautifulSoup
import random
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Dashboard Klasifikasi Berita", page_icon="📰", layout="wide")

# --- FUNGSI-FUNGSI HELPER ---

@st.cache_data
def load_local_csv(file_path):
    """Membaca file CSV, mengonversi tanggal dengan aman, dan melakukan cache."""
    try:
        df = pd.read_csv(file_path, delimiter=';')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.dropna(subset=['datetime'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan: `{file_path}`")
        return None
    except Exception as e:
        st.error(f"Terjadi error tak terduga saat memuat CSV: {e}")
        return None

@st.cache_resource
def load_classification_model():
    """Memuat model klasifikasi dan vectorizer dari file."""
    try:
        model_path = os.path.join('model_svc', 'linear_svc_model.joblib')
        vectorizer_path = os.path.join('model_svc', 'tfidf_vectorizer.joblib')
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_resource
def create_stopword_remover():
    """Membuat objek stopword remover sekali saja."""
    factory = StopWordRemoverFactory()
    return factory.create_stop_word_remover()

def preprocess_text(text, remover):
    """Membersihkan teks input untuk prediksi menggunakan remover yang sudah ada."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return remover.remove(text)

def scrape_kontan_article(url):
    """Mengambil teks artikel dari URL kontan.co.id."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        candidates = ["div.detail-konten", "div.article-content", "div.contents", "div.konten-artikel", "div#content", "div.main-content", "article", "main"]
        for selector in candidates:
            container = soup.select_one(selector)
            if container:
                paragraphs = container.find_all("p")
                text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
                if len(text) > 200: return text
        fallback_paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in fallback_paragraphs if p.get_text(strip=True))
        return text if len(text) > 200 else None
    except Exception as e:
        print(f"[Scraping Error]: {e}")
        return None

def display_prediction_result(text_input, model, vec, remover):
    """Fungsi untuk memproses dan menampilkan hasil prediksi."""
    with st.spinner("Memprediksi topik berita..."):
        cleaned = preprocess_text(text_input, remover)
        vectorized = vec.transform([cleaned])
        prediction = model.predict(vectorized)
        st.success(f"## 🏷️ Topik Prediksi: {prediction[0]}")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Probabilitas Prediksi")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vectorized)
                df_proba = pd.DataFrame(proba[0], index=model.classes_, columns=["Probabilitas"]).reset_index()
                df_proba.rename(columns={"index": "Topik"}, inplace=True)
                fig = px.bar(df_proba, x="Probabilitas", y="Topik", orientation="h", text="Probabilitas")
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                fig.update_layout(yaxis_title="", xaxis_title="Probabilitas", yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("🔍 WordCloud Kata Kunci")
            try:
                wc = WordCloud(width=800, height=400, background_color='white', colormap='cividis').generate(cleaned)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except ValueError:
                st.warning("Teks terlalu pendek untuk membuat WordCloud.")

# --- PEMUATAN OBJEK GLOBAL DI AWAL ---
stopword_remover = create_stopword_remover()
ml_model, vectorizer = load_classification_model()
DATASET_PATH = 'data_sintetis_topic_news_kontan.csv'
df_dashboard = load_local_csv(DATASET_PATH)

if df_dashboard is not None and 'article' in df_dashboard.columns:
    df_dashboard['panjang_artikel'] = df_dashboard['article'].str.len().fillna(0)

# --- PERBAIKAN: Inisialisasi variabel filter di awal ---
start_date, end_date, selected_topics = None, None, []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200)
    st.header("Pengaturan Dasbor")
    dashboard_title = st.text_input("Judul Dasbor", "Analisis Berita Kontan")
    st.markdown("---")
    st.header("Sumber Data")
    if df_dashboard is not None:
        st.success(f"Berhasil memuat data dari:\n`{DATASET_PATH}`")
        st.header("Filter Data")
        if 'datetime' in df_dashboard.columns and not df_dashboard['datetime'].isnull().all():
            min_date = df_dashboard['datetime'].min().date()
            max_date = df_dashboard['datetime'].max().date()
            date_range = st.date_input("Pilih Rentang Tanggal", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            start_date, end_date = (date_range if isinstance(date_range, tuple) and len(date_range) == 2 else (min_date, max_date))
        
        all_topics = df_dashboard['label_topic'].unique()
        selected_topics = st.multiselect("Filter berdasarkan Topik:", options=all_topics, default=list(all_topics))

# --- FILTER DATA ---
df_filtered = pd.DataFrame() # Buat dataframe kosong sebagai default
if df_dashboard is not None:
    df_to_filter = df_dashboard.copy()
    if start_date and end_date and 'datetime' in df_to_filter.columns:
        date_mask = (df_to_filter['datetime'].dt.date >= start_date) & (df_to_filter['datetime'].dt.date <= end_date)
        df_to_filter = df_to_filter.loc[date_mask]
    if selected_topics:
        df_filtered = df_to_filter[df_to_filter['label_topic'].isin(selected_topics)].copy()
    else:
        df_filtered = df_to_filter.copy()

# --- HALAMAN UTAMA ---
st.title(f"📊 {dashboard_title}")
if start_date and end_date:
    st.markdown(f"Menampilkan analisis dari **{start_date.strftime('%d %b %Y')}** hingga **{end_date.strftime('%d %b %Y')}**")
st.markdown("---")

tab1, tab2 = st.tabs(["📈 Dasbor Visual", "🤖 Klasifikasi Artikel"])

# --- KONTEN TAB 1: DASBOR VISUAL ---
with tab1:
    if not df_filtered.empty:
        try:
            total_artikel, jumlah_topik = len(df_filtered), df_filtered['label_topic'].nunique()
            rata_rata_panjang_artikel = int(df_filtered['panjang_artikel'].mean())
            col1, col2, col3 = st.columns(3)
            col1.metric(label="Total Artikel (Filter)", value=f"{total_artikel:,}")
            col2.metric(label="Jumlah Topik Berbeda", value=f"{jumlah_topik:,}")
            col3.metric(label="Rata-rata Panjang Artikel", value=f"{rata_rata_panjang_artikel:,} karakter")
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Jumlah Artikel per Topik")
                topic_counts = df_filtered['label_topic'].value_counts().reset_index()
                fig_bar = px.bar(topic_counts, x='count', y='label_topic', orientation='h', title="Distribusi Artikel")
                fig_bar.update_layout(yaxis_title="Topik", xaxis_title="Jumlah Artikel")
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_chart2:
                st.subheader("Distribusi Topik Populer")
                pie_counts = df_filtered['label_topic'].value_counts().reset_index()
                fig_pie = px.pie(pie_counts, names='label_topic', values='count', title="Persentase Topik")
                st.plotly_chart(fig_pie, use_container_width=True)
            st.subheader("Detail Data (Filter)")
            st.dataframe(df_filtered)
        except KeyError as e:
            st.error(f"Nama kolom salah: {e}. Pastikan file CSV Anda memiliki kolom 'label_topic' dan 'article'.")
    else:
        st.info("Tidak ada data untuk ditampilkan pada filter yang dipilih.")

# --- KONTEN TAB 2: KLASIFIKASI ARTIKEL ---
with tab2:
    st.header("Klasifikasikan Topik Artikel Baru")
    input_manual, input_url = st.tabs(["✍️ Tempel Teks Manual", "🔗 Scrape dari URL"])
    if ml_model and vectorizer:
        with input_manual:
            st.subheader("Tempelkan Teks Artikel")
            def clear_text_manual(): st.session_state.artikel_input_manual = ""
            user_input_manual = st.text_area("Teks Artikel:", height=250, key="artikel_input_manual", label_visibility="collapsed")
            col1_manual, col2_manual = st.columns(2)
            with col1_manual:
                predict_button_manual = st.button("Prediksi Teks", type="primary", use_container_width=True)
            with col2_manual:
                st.button("Hapus Teks", on_click=clear_text_manual, use_container_width=True)
            if predict_button_manual and user_input_manual:
                display_prediction_result(user_input_manual, ml_model, vectorizer, stopword_remover)
        with input_url:
            st.subheader("Masukkan URL Artikel dari Kontan.co.id")
            url_input = st.text_input("URL Artikel", placeholder="https://investasi.kontan.co.id/...")
            if st.button("Scrape & Prediksi", type="primary"):
                if url_input:
                    with st.spinner(f"Mengambil data dari {url_input}..."):
                        article_text = scrape_kontan_article(url_input)
                    if article_text:
                        st.success("Berhasil mengambil teks artikel!")
                        with st.expander("Lihat Teks yang Di-scrape"):
                            st.write(article_text)
                        display_prediction_result(article_text, ml_model, vectorizer, stopword_remover)
                    else:
                        st.error("Gagal mengambil teks dari URL. Pastikan URL valid dan struktur website tidak berubah.")
                else:
                    st.warning("Harap masukkan URL terlebih dahulu.")
    else:
        st.error("Model klasifikasi tidak ditemukan. Periksa folder 'model_svc'.")