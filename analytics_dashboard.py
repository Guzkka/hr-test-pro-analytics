import streamlit as st
import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================
st.set_page_config(
    page_title="HR Test Pro - Аналитика",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 HR Test Pro - Глубокая аналитика")
st.markdown("---")

# ============================================
# БОКОВАЯ ПАНЕЛЬ - ВИДЖЕТЫ (вне кэша!)
# ============================================
st.sidebar.header("📁 Источник данных")
data_source = st.sidebar.radio(
    "Выберите источник:",
    ["🔥 Firebase Firestore (реальные данные)", "📊 Kaggle датасет (fair_recrutment_dataset)"]
)

# Виджет загрузки файла (вне кэша!)
uploaded_file = None
if "Kaggle" in data_source:
    uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])

# ============================================
# FIREBASE ИНИЦИАЛИЗАЦИЯ
# ============================================
@st.cache_resource
def init_firebase():
    """Инициализация Firebase Admin SDK"""
    try:
        if not firebase_admin._apps:
            cred_dict = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
            }
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"❌ Ошибка подключения к Firebase: {e}")
        return None

# ============================================
# ЗАГРУЗКА ИЗ FIRESTORE (кэшируется)
# ============================================
@st.cache_data(ttl=300)
def load_from_firebase():
    """Загрузка реальных данных из Firestore"""
    db = init_firebase()
    if not db:
        return None
    
    try:
        results = []
        for doc in db.collection('results').stream():
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
        
        if not results:
            return None
        
        df = pd.DataFrame(results)
        
        processed = []
        for _, row in df.iterrows():
            match_score = row.get('matchScore', np.random.normal(65, 15))
            test_score = row.get('score', row.get('percentage', np.random.normal(68, 18)))
            
            processed.append({
                'matchScore': float(match_score),
                'testScore': float(test_score),
                'attemptCount': int(row.get('attemptNumber', 1)),
                'responseTime_days': 2.5,
                'target_success': 1 if float(test_score) >= 70 else 0,
                'candidateName': row.get('candidateName', 'Аноним'),
                'testTitle': row.get('testTitle', 'Без названия'),
                'completedAt': row.get('completedAt')
            })
        
        return pd.DataFrame(processed)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки из Firestore: {e}")
        return None

# ============================================
# ЗАГРУЗКА KAGGLE ДАТАСЕТА (кэшируется)
# ============================================
@st.cache_data
def load_kaggle_from_file(uploaded_file):
    """Загрузка датасета из загруженного файла"""
    if uploaded_file is None:
        return None
    
    try:
        df = pd.read_csv(uploaded_file)
        
        df.rename(columns={
            'Skill_Score': 'matchScore',
            'Technical_Test_Score': 'testScore',
            'Certifications_Count': 'attemptCount',
            'Hiring_Decision': 'target_success'
        }, inplace=True)
        
        if 'responseTime_days' not in df.columns:
            df['responseTime_days'] = np.random.exponential(2.5, len(df)).clip(0.5, 14).round(1)
        
        required_cols = ['matchScore', 'testScore', 'attemptCount', 'responseTime_days', 'target_success']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"❌ В датасете отсутствует колонка: {col}")
                return None
        
        df = df[required_cols].copy()
        df = df.dropna()
        
        df['matchScore'] = pd.to_numeric(df['matchScore'], errors='coerce').fillna(65)
        df['testScore'] = pd.to_numeric(df['testScore'], errors='coerce').fillna(68)
        df['attemptCount'] = pd.to_numeric(df['attemptCount'], errors='coerce').fillna(1).astype(int)
        
        return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки Kaggle датасета: {e}")
        return None

# ============================================
# ЗАГРУЗКА ДАННЫХ (главная логика)
# ============================================
df = None

if "Firebase" in data_source:
    with st.spinner("🔄 Подключение к Firebase..."):
        df = load_from_firebase()
        if df is not None:
            st.sidebar.success(f"✅ Загружено {len(df)} записей из Firestore")
            st.sidebar.info(f"📊 Успешных: {df['target_success'].sum()} ({df['target_success'].mean()*100:.1f}%)")
        else:
            st.sidebar.warning("⚠️ Нет данных в Firestore или ошибка подключения")

else:
    if uploaded_file is not None:
        with st.spinner("🔄 Загрузка файла..."):
            df = load_kaggle_from_file(uploaded_file)
            if df is not None:
                st.sidebar.success(f"✅ Загружено {len(df):,} записей")
                st.sidebar.info(f"📊 Успешных наймов: {df['target_success'].sum():,} ({df['target_success'].mean()*100:.1f}%)")
    else:
        st.sidebar.info("📁 Загрузите CSV файл для анализа")

# ============================================
# ОСНОВНОЙ ДАШБОРД
# ============================================
if df is not None and len(df) > 0:
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Всего кандидатов", f"{len(df):,}")
    with col2:
        st.metric("📊 Средний MatchScore", f"{df['matchScore'].mean():.1f}%")
    with col3:
        st.metric("📈 Средний TestScore", f"{df['testScore'].mean():.1f}%")
    with col4:
        success_rate = df['target_success'].mean() * 100
        st.metric("✅ Успешных", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Распределение", 
        "🎯 ML Прогнозирование", 
        "🔍 Кластеризация",
        "📅 Сезонность",
        "💡 Рекомендации"
    ])
    
    # ===== ВКЛАДКА 1: РАСПРЕДЕЛЕНИЕ =====
    with tab1:
        st.subheader("Распределение показателей")
        col1, col2 = st.columns(2)
        with col1:
            fig_match = px.histogram(df, x='matchScore', nbins=30, 
                                    title='Распределение MatchScore',
                                    labels={'matchScore': 'Match Score (%)'})
            st.plotly_chart(fig_match, use_container_width=True)
        with col2:
            fig_test = px.histogram(df, x='testScore', nbins=30,
                                   title='Распределение TestScore',
                                   labels={'testScore': 'Test Score (%)'})
            st.plotly_chart(fig_test, use_container_width=True)
    
    # ===== ВКЛАДКА 2: ML ПРОГНОЗИРОВАНИЕ =====
    with tab2:
        st.subheader("🎯 ML Прогнозирование успешности")
        X = df[['matchScore', 'testScore', 'attemptCount', 'responseTime_days']]
        y = df['target_success']
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Признак': X.columns,
            'Важность': model.feature_importances_
        }).sort_values('Важность', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_imp = px.bar(importance_df, x='Важность', y='Признак',
                           title='Важность признаков', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        with col2:
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            st.metric("AUC-ROC", f"{roc_auc_score(y, y_pred_proba):.3f}")
            st.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
            st.info(f"💡 MatchScore даёт {importance_df.iloc[0]['Важность']*100:.1f}% вклада в прогноз")
    
    # ===== ВКЛАДКА 3: КЛАСТЕРИЗАЦИЯ =====
    with tab3:
        st.subheader("🔍 Кластеризация кандидатов")
        X_cluster = df[['matchScore', 'attemptCount']].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        k_optimal = st.slider("Количество кластеров:", 2, 6, 4)
        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_clusters = px.scatter(df, x='matchScore', y='attemptCount',
                                     color='cluster', title='Кластеры кандидатов',
                                     labels={'matchScore': 'Match Score', 
                                            'attemptCount': 'Количество попыток'})
            st.plotly_chart(fig_clusters, use_container_width=True)
        with col2:
            st.write("**Статистика по кластерам:**")
            cluster_stats = df.groupby('cluster').agg({
                'matchScore': 'mean',
                'testScore': 'mean',
                'attemptCount': 'mean',
                'target_success': 'mean'
            }).round(2)
            st.dataframe(cluster_stats)
    
    # ===== ВКЛАДКА 4: СЕЗОННОСТЬ =====
    with tab4:
        st.subheader("📅 Анализ сезонности")
        if 'completedAt' in df.columns and df['completedAt'].notna().any():
            df['month'] = pd.to_datetime(df['completedAt'], errors='coerce').dt.month
            monthly_counts = df.groupby('month').size()
            fig_season = px.bar(x=monthly_counts.index, y=monthly_counts.values,
                               labels={'x': 'Месяц', 'y': 'Количество откликов'},
                               title='Сезонность откликов')
            st.plotly_chart(fig_season, use_container_width=True)
        else:
            st.info("ℹ️ Нет данных о дате для анализа сезонности")
    
    # ===== ВКЛАДКА 5: РЕКОМЕНДАЦИИ =====
    with tab5:
        st.subheader("💡 Рекомендации для HR")
        top_candidates = df[df['target_success'] == 1]
        
        st.write("**Профиль успешного кандидата:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Средний MatchScore", f"{top_candidates['matchScore'].mean():.1f}%")
            st.metric("Средний TestScore", f"{top_candidates['testScore'].mean():.1f}%")
        with col2:
            st.metric("Среднее время отклика", f"{top_candidates['responseTime_days'].mean():.1f} дн.")
            st.metric("Среднее количество попыток", f"{top_candidates['attemptCount'].mean():.1f}")
        
        st.markdown("---")
        st.write("**Конкретные рекомендации:**")
        match_threshold = top_candidates['matchScore'].quantile(0.25)
        st.write(f"✅ **Автоматически рекомендовать оффер** при MatchScore > {match_threshold:.0f}%")
        st.write(f"⚠️ **Требует собеседования** при MatchScore {match_threshold*.7:.0f}-{match_threshold:.0f}%")
        st.write(f"❌ **Отказывать** при MatchScore < {match_threshold*.7:.0f}%")

else:
    st.warning("⚠️ Нет данных для анализа. Выберите источник данных в боковой панели.")

st.markdown("---")
st.markdown("*HR Test Pro Analytics Dashboard v1.0*")
