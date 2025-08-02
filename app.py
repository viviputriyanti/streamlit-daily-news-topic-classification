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
    # Sekarang fungsi ini bisa melihat PREDICTED_DATA_PATH
    if not os.path.exists(PREDICTED_DATA_PATH):
        df = pd.DataFrame(columns=['judul', 'url', 'content', 'tag', 'datetime', 'article', 'label_topic'])
        df.to_csv(PREDICTED_DATA_PATH, index=False, sep=';')

@st.cache_data(ttl=30)
def load_combined_data():
    # Sekarang fungsi ini bisa melihat kedua variabel PATH
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
        
        combined_stopwords = set(sastrawi_stopwords).union(custom_stopwords)
        
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
            prediction_input = title + ". " + original_article_text
            prediction, proba_df = predict_topic_and_proba(prediction_input, ml_model, vectorizer, stopword_remover)
            st.success(f"**Prediksi Topik Utama:** `{prediction}`")
            with st.expander("Lihat Detail Probabilitas Prediksi"):
                st.dataframe(proba_df, use_container_width=True)
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
        possible_selectors = ['.detail-desk.pr-3', 'div.tm-content', 'div.detail_desk', '#content']
        article_text = ''
        for selector in possible_selectors:
            content_div = soup.select_one(selector)
            if content_div:
                paragraphs = content_div.find_all('p')
                article_text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                if len(article_text.split()) > 20: return article_text
        if not article_text:
            all_paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text(strip=True) for p in all_paragraphs])
        return article_text if article_text else None
    except requests.RequestException:
        return None

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
        
        if 'select_all' not in st.session_state:
            st.session_state.select_all = True

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
        col1.metric("Total Artikel", f"{len(df_filtered)} 📈")
        col2.metric("Total Topik", f"{df_filtered['label_topic'].nunique()} 📚")
        st.markdown("<hr>", unsafe_allow_html=True)

        viz_col1, viz_col2 = st.columns([1, 1.2])
        with viz_col1:
            st.subheader("Distribusi Topik")
            topic_counts = df_filtered['label_topic'].value_counts()
            fig_pie = px.pie(topic_counts, names=topic_counts.index, values=topic_counts.values, hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
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
        df_display['artikel'] = df_display['judul'].fillna('') + ". " + df_display['content'].fillna('')
        df_display['datetime'] = df_display['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_display[['datetime', 'artikel', 'label_topic', 'url']], use_container_width=True)
    else:
        st.warning("Tidak ada data yang cocok dengan filter yang Anda pilih.")

# --- HALAMAN KLASIFIKASI ---
elif page == "Klasifikasi Teks":
    st.title("KLASIFIKASI TOPIK BERITA")
    if ml_model:
        input_manual, input_url, input_csv = st.tabs(["📝 Input Manual", "🔗 Scrape dari URL", "📂 Unggah File CSV"])
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
                    title_scrape = url_input.split('/')[-1].replace('-', ' ').capitalize()
                    article_text_scrape = scrape_kontan_article(url_input)
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
                        if st.button("Prediksi", type="primary"):
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