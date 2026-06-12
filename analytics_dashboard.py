import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Конфигурация страницы
st.set_page_config(
    page_title="HR Test Pro - Аналитика",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок
st.title("📊 HR Test Pro - Глубокая аналитика")
st.markdown("---")

# Боковая панель для загрузки данных
st.sidebar.header("📁 Загрузка данных")
upload_method = st.sidebar.radio(
    "Источник данных:",
    ["Загрузить CSV", "Использовать демо-данные"]
)

# Функция загрузки данных
@st.cache_data
def load_data():
    if upload_method == "Загрузить CSV":
        uploaded_file = st.sidebar.file_uploader("Загрузите CSV файл", type=['csv'])
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        else:
            st.sidebar.warning("Загрузите файл для анализа")
            return None
    else:
        # Генерация демо-данных (как в Colab)
        np.random.seed(42)
        n_samples = 2000
        return pd.DataFrame({
            'matchScore': np.random.normal(65, 15, n_samples).clip(20, 98).round(1),
            'testScore': np.random.normal(68, 18, n_samples).clip(30, 100).round(1),
            'attemptCount': np.random.choice([1, 2, 3], n_samples, p=[0.65, 0.25, 0.10]),
            'responseTime_days': np.random.exponential(2.5, n_samples).clip(0.5, 14).round(1),
            'target_success': np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
        })

df = load_data()

if df is not None:
    # Основные метрики
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего кандидатов", len(df))
    with col2:
        st.metric("Средний MatchScore", f"{df['matchScore'].mean():.1f}%")
    with col3:
        st.metric("Средний TestScore", f"{df['testScore'].mean():.1f}%")
    with col4:
        success_rate = df['target_success'].mean() * 100
        st.metric("Успешных кандидатов", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Вкладки для разных видов аналитики
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Распределение", 
        "🎯 ML Прогнозирование", 
        "🔍 Кластеризация",
        "📅 Сезонность",
        "💡 Рекомендации"
    ])
    
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
    
    with tab2:
        st.subheader(" ML Прогнозирование успешности")
        
        # Подготовка данных
        X = df[['matchScore', 'testScore', 'attemptCount', 'responseTime_days']]
        y = df['target_success']
        
        # Обучение модели
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Важность признаков
        importance_df = pd.DataFrame({
            'Признак': X.columns,
            'Важность': model.feature_importances_
        }).sort_values('Важность', ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_imp = px.bar(importance_df, x='Важность', y='Признак',
                           title='Важность признаков',
                           orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            # Метрики модели
            from sklearn.metrics import roc_auc_score, accuracy_score
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            st.metric("AUC-ROC", f"{roc_auc_score(y, y_pred_proba):.3f}")
            st.metric("Accuracy", f"{accuracy_score(y, y_pred):.3f}")
            
            st.info(f"💡 MatchScore даёт {importance_df.iloc[0]['Важность']*100:.1f}% вклада в прогноз")
    
    with tab3:
        st.subheader("🔍 Кластеризация кандидатов")
        
        # Кластеризация
        X_cluster = df[['matchScore', 'attemptCount']].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cluster)
        
        # Метод локтя для выбора k
        inertia = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)
        
        k_optimal = st.slider("Количество кластеров:", 2, 6, 4)
        
        kmeans = KMeans(n_clusters=k_optimal, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_clusters = px.scatter(df, x='matchScore', y='attemptCount',
                                     color='cluster', 
                                     title='Кластеры кандидатов',
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
    
    with tab4:
        st.subheader("📅 Анализ сезонности")
        
        # Генерация временных данных (если нет в датасете)
        if 'appliedAt' not in df.columns:
            dates = pd.date_range(start='2025-01-01', periods=len(df), freq='D')
            df['appliedAt'] = dates
        
        df['month'] = pd.to_datetime(df['appliedAt']).dt.month
        monthly_counts = df.groupby('month').size()
        
        fig_season = px.bar(x=monthly_counts.index, y=monthly_counts.values,
                           labels={'x': 'Месяц', 'y': 'Количество откликов'},
                           title='Сезонность откликов')
        st.plotly_chart(fig_season, use_container_width=True)
    
    with tab5:
        st.subheader("💡 Рекомендации для HR")
        
        # Анализ лучших кандидатов
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
        st.write(f"⚠️ **Требует дополнительного собеседования** при MatchScore {match_threshold*.7:.0f}-{match_threshold:.0f}%")
        st.write(f"❌ **Отказывать** при MatchScore < {match_threshold*.7:.0f}%")
        
        st.info("💡 Персонализируйте подход по кластерам для повышения конверсии")

else:
    st.warning("⚠️ Нет данных для анализа. Загрузите CSV файл или выберите демо-данные.")

# Футер
st.markdown("---")
st.markdown("*HR Test Pro Analytics Dashboard v1.0*")