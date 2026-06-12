import streamlit as st
import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

st.set_page_config(page_title="HR Test Pro - Аналитика", layout="wide", page_icon="📊")

st.title("📊 HR Test Pro - Глубокая аналитика")
st.markdown("---")

# ============================================
# FIREBASE ИНИЦИАЛИЗАЦИЯ
# ============================================
@st.cache_resource
def init_firebase():
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
# ЗАГРУЗКА ИЗ FIRESTORE
# ============================================
@st.cache_data(ttl=300)
def load_from_firebase():
    db = init_firebase()
    if not db:
        return None, None
    
    try:
        results = []
        for doc in db.collection('results').stream():
            data = doc.to_dict()
            data['id'] = doc.id
            results.append(data)
        
        if not results:
            return None, None
        
        df = pd.DataFrame(results)
        
        processed = []
        for _, row in df.iterrows():
            match_score = row.get('matchScore', np.random.normal(65, 15))
            test_score = row.get('score', row.get('percentage', np.random.normal(68, 18)))
            
            processed.append({
                'matchScore': float(match_score) if not pd.isna(match_score) else 65.0,
                'testScore': float(test_score) if not pd.isna(test_score) else 68.0,
                'attemptCount': int(row.get('attemptNumber', 1)) if not pd.isna(row.get('attemptNumber')) else 1,
                'responseTime_days': 2.5,
                'target_success': 1 if float(test_score) >= 70 else 0,
                'candidateName': row.get('candidateName', 'Аноним'),
                'testTitle': row.get('testTitle', 'Без названия'),
                'completedAt': row.get('completedAt')
            })
        
        df_processed = pd.DataFrame(processed)
        dates = pd.to_datetime(df_processed['completedAt'], errors='coerce').dropna()
        return df_processed, dates
    except Exception as e:
        st.error(f"❌ Ошибка загрузки из Firestore: {e}")
        return None, None

# ============================================
# ЗАГРУЗКА KAGGLE ДАТАСЕТА
# ============================================
@st.cache_data
def load_kaggle_dataset():
    file_path = 'fair_recrutment_dataset final.csv'
    
    if not os.path.exists(file_path):
        return None, None
    
    try:
        df = pd.read_csv(file_path)
        
        df.rename(columns={
            'Skill_Score': 'matchScore',
            'Technical_Test_Score': 'testScore',
            'Certifications_Count': 'attemptCount',
            'Hiring_Decision': 'target_success'
        }, inplace=True)
        
        if 'responseTime_days' not in df.columns:
            df['responseTime_days'] = 0
        
        required_cols = ['matchScore', 'testScore', 'attemptCount', 'responseTime_days', 'target_success']
        for col in required_cols:
            if col not in df.columns:
                return None, None
        
        df = df[required_cols].copy()
        df = df.fillna(0)
        
        df['matchScore'] = pd.to_numeric(df['matchScore'], errors='coerce').fillna(0)
        df['testScore'] = pd.to_numeric(df['testScore'], errors='coerce').fillna(0)
        df['attemptCount'] = pd.to_numeric(df['attemptCount'], errors='coerce').fillna(0)
        
        return df, None
    except Exception as e:
        st.error(f"❌ Ошибка загрузки Kaggle датасета: {e}")
        return None, None

# ============================================
# БОКОВАЯ ПАНЕЛЬ
# ============================================
st.sidebar.header("📁 Источник данных")
data_source = st.sidebar.radio(
    "Выберите источник:",
    [" Firebase Firestore", "📊 Kaggle датасет"]
)

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================
df = None
dates = None

if "Firebase" in data_source:
    with st.spinner("🔄 Подключение к Firebase..."):
        df, dates = load_from_firebase()
        if df is not None:
            st.sidebar.success(f"✅ Загружено {len(df)} записей из Firestore")
            st.sidebar.info(f"📊 Успешных: {df['target_success'].sum()} ({df['target_success'].mean()*100:.1f}%)")
else:
    with st.spinner("🔄 Загрузка Kaggle датасета..."):
        df, dates = load_kaggle_dataset()
        if df is not None:
            st.sidebar.success(f"✅ Загружено {len(df):,} записей из Kaggle")
            st.sidebar.info(f"📊 Успешных наймов: {df['target_success'].sum():,} ({df['target_success'].mean()*100:.1f}%)")

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
    
    # Подготовка данных для ML
    feature_columns = ['matchScore', 'testScore', 'attemptCount', 'responseTime_days']
    X = df[feature_columns].copy()
    y = df['target_success'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение моделей
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Метрики
    def get_metrics(y_true, y_pred, y_pred_proba):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1-Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_true, y_pred_proba)
        }
    
    metrics_lr = get_metrics(y_test, y_pred_lr, y_pred_proba_lr)
    metrics_rf = get_metrics(y_test, y_pred_rf, y_pred_proba_rf)
    
    # Вкладки
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Метрики моделей",
        "🎯 ROC-кривые",
        "📈 Важность признаков",
        "🔍 Кластеризация",
        "📅 Сезонность",
        "📋 Матрицы ошибок"
    ])
    
    # ===== ВКЛАДКА 1: МЕТРИКИ МОДЕЛЕЙ =====
    with tab1:
        st.subheader("Сравнение моделей машинного обучения")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression")
            st.metric("Accuracy", f"{metrics_lr['Accuracy']:.3f}")
            st.metric("Precision", f"{metrics_lr['Precision']:.3f}")
            st.metric("Recall", f"{metrics_lr['Recall']:.3f}")
            st.metric("F1-Score", f"{metrics_lr['F1-Score']:.3f}")
            st.metric("ROC-AUC", f"{metrics_lr['ROC-AUC']:.3f}")
        
        with col2:
            st.subheader("Random Forest")
            st.metric("Accuracy", f"{metrics_rf['Accuracy']:.3f}")
            st.metric("Precision", f"{metrics_rf['Precision']:.3f}")
            st.metric("Recall", f"{metrics_rf['Recall']:.3f}")
            st.metric("F1-Score", f"{metrics_rf['F1-Score']:.3f}")
            st.metric("ROC-AUC", f"{metrics_rf['ROC-AUC']:.3f}")
        
        st.markdown("---")
        st.info(f" **Лучшая модель:** {'Random Forest' if metrics_rf['ROC-AUC'] > metrics_lr['ROC-AUC'] else 'Logistic Regression'} (AUC = {max(metrics_lr['ROC-AUC'], metrics_rf['ROC-AUC']):.3f})")
    
    # ===== ВКЛАДКА 2: ROC-КРИВЫЕ =====
    with tab2:
        st.subheader("ROC-кривые для моделей прогнозирования")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
        ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {metrics_lr["ROC-AUC"]:.3f})', linewidth=2)
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
        ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {metrics_rf["ROC-AUC"]:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-кривые')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # ===== ВКЛАДКА 3: ВАЖНОСТЬ ПРИЗНАКОВ =====
    with tab3:
        st.subheader("Важность признаков (Random Forest)")
        
        feature_importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': rf.feature_importances_
        }).sort_values('Важность', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_importance)))
        ax.barh(feature_importance['Признак'], feature_importance['Важность'], color=colors, edgecolor='black')
        ax.set_xlabel('Важность')
        ax.set_title('Важность признаков')
        ax.invert_yaxis()
        
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("**Ранжирование признаков:**")
        for _, row in feature_importance.iterrows():
            bar = '█' * int(row['Важность'] * 50)
            st.write(f"{row['Признак']}: {bar} {row['Важность']*100:.1f}%")
    
    # ===== ВКЛАДКА 4: КЛАСТЕРИЗАЦИЯ =====
    with tab4:
        st.subheader("Кластеризация кандидатов (K-means)")
        
        cluster_features = ['matchScore', 'testScore', 'attemptCount']
        X_cluster = df[cluster_features].copy().fillna(0)
        
        scaler_cluster = StandardScaler()
        X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
        
        # Метод локтя
        inertias = []
        for k in range(1, 8):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_cluster_scaled)
            inertias.append(kmeans.inertia_)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(range(1, 8), inertias, 'bo-')
        axes[0].set_xlabel('Количество кластеров (k)')
        axes[0].set_ylabel('Инерция')
        axes[0].set_title('Метод "локтя" для выбора k')
        axes[0].grid(True, alpha=0.3)
        
        k_optimal = st.slider("Количество кластеров", 2, 6, 4)
        
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_cluster_scaled)
        
        scatter = axes[1].scatter(df['matchScore'], df['testScore'],
                                  c=df['cluster'], cmap='viridis', alpha=0.6)
        axes[1].set_xlabel('MatchScore (%)')
        axes[1].set_ylabel('TestScore (%)')
        axes[1].set_title(f'Кластеры кандидатов (k={k_optimal})')
        plt.colorbar(scatter, ax=axes[1], label='Кластер')
        
        st.pyplot(fig)
        
        st.markdown("---")
        st.write("**Статистика по кластерам:**")
        cluster_stats = df.groupby('cluster').agg({
            'matchScore': 'mean',
            'testScore': 'mean',
            'attemptCount': 'mean',
            'target_success': 'mean'
        }).round(2)
        cluster_stats.columns = ['Ср. MatchScore', 'Ср. TestScore', 'Ср. попыток', 'Доля успешных']
        st.dataframe(cluster_stats)
    
    # ===== ВКЛАДКА 5: СЕЗОННОСТЬ =====
    with tab5:
        st.subheader("Анализ сезонности откликов")
        
        all_dates = []
        
        if dates is not None and len(dates) > 0:
            all_dates.extend(dates.tolist())
            st.info(f" Реальных дат: {len(dates)}")
        
        if len(all_dates) < 30:
            st.warning(f"⚠️ Реальных дат мало ({len(all_dates)}). Добавьте больше данных для анализа сезонности.")
            st.info("💡 Сезонность будет доступна при загрузке данных с датами откликов")
        else:
            df_all_dates = pd.DataFrame({'appliedAt': all_dates})
            df_all_dates['appliedAt'] = pd.to_datetime(df_all_dates['appliedAt'], errors='coerce')
            
            if df_all_dates['appliedAt'].dt.tz is not None:
                df_all_dates['appliedAt'] = df_all_dates['appliedAt'].dt.tz_localize(None)
            
            df_all_dates['month'] = df_all_dates['appliedAt'].dt.month
            df_all_dates['day_of_week'] = df_all_dates['appliedAt'].dt.dayofweek
            df_all_dates['hour'] = df_all_dates['appliedAt'].dt.hour
            
            monthly_counts = df_all_dates.groupby('month').size().reindex(range(1, 13), fill_value=0)
            avg_monthly = monthly_counts.mean()
            seasonal_coef = (monthly_counts / avg_monthly).round(2)
            
            months_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
            days_ru = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            axes[0, 0].bar(months_ru, monthly_counts.values, color='steelblue', edgecolor='black')
            axes[0, 0].set_xlabel('Месяц')
            axes[0, 0].set_ylabel('Количество откликов')
            axes[0, 0].set_title('Сезонность по месяцам')
            
            colors_coef = ['#2ecc71' if c > 1.1 else '#e74c3c' if c < 0.9 else '#95a5a6' for c in seasonal_coef]
            axes[0, 1].bar(months_ru, seasonal_coef.values, color=colors_coef, edgecolor='black')
            axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Месяц')
            axes[0, 1].set_ylabel('Сезонный коэффициент')
            axes[0, 1].set_title('Сезонные коэффициенты')
            
            weekday_counts = df_all_dates.groupby('day_of_week').size().reindex(range(0, 7), fill_value=0)
            axes[1, 0].bar(days_ru, weekday_counts.values, color='coral', edgecolor='black')
            axes[1, 0].set_xlabel('День недели')
            axes[1, 0].set_ylabel('Количество откликов')
            axes[1, 0].set_title('Распределение по дням недели')
            
            hour_counts = df_all_dates.groupby('hour').size().reindex(range(0, 24), fill_value=0)
            axes[1, 1].bar(range(24), hour_counts.values, color='seagreen', edgecolor='black')
            axes[1, 1].set_xlabel('Час дня')
            axes[1, 1].set_ylabel('Количество откликов')
            axes[1, 1].set_title('Распределение по часам')
            axes[1, 1].set_xticks(range(0, 24, 3))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("---")
            st.write("**Сезонные коэффициенты:**")
            for i in range(12):
                coef = seasonal_coef.iloc[i]
                trend = "🔺 ВЫСОКИЙ" if coef > 1.15 else "🔻 НИЗКИЙ" if coef < 0.85 else "➖ СРЕДНИЙ"
                st.write(f"{months_ru[i]}: {coef:.2f} - {trend}")
    
    # ===== ВКЛАДКА 6: МАТРИЦЫ ОШИБОК =====
    with tab6:
        st.subheader("Матрицы ошибок (Confusion Matrices)")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Logistic Regression\nAccuracy: {metrics_lr["Accuracy"]:.3f}')
        axes[0].set_xlabel('Предсказано')
        axes[0].set_ylabel('Факт')
        
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f'Random Forest\nAccuracy: {metrics_rf["Accuracy"]:.3f}')
        axes[1].set_xlabel('Предсказано')
        axes[1].set_ylabel('Факт')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        st.info("💡 **Интерпретация:** Диагональные элементы - правильные предсказания, остальные - ошибки классификации")

else:
    st.warning("⚠️ Нет данных для анализа. Выберите источник данных в боковой панели.")

st.markdown("---")
st.markdown("*HR Test Pro Analytics Dashboard v1.0*")
