import streamlit as st
import pandas as pd
import numpy as np
import os
import firebase_admin
from firebase_admin import credentials, firestore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

st.set_page_config(page_title="HR Test Pro - Аналитика", layout="wide")
st.title("📊 HR Test Pro - Глубокая аналитика")

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
# ЗАГРУЗКА ИЗ FIRESTORE
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
                'matchScore': float(match_score) if not pd.isna(match_score) else 65.0,
                'testScore': float(test_score) if not pd.isna(test_score) else 68.0,
                'attemptCount': int(row.get('attemptNumber', 1)) if not pd.isna(row.get('attemptNumber')) else 1,
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
# ЗАГРУЗКА KAGGLE ДАТАСЕТА
# ============================================
@st.cache_data
def load_kaggle_dataset():
    """Загрузка датасета Kaggle"""
    file_path = 'fair_recrutment_dataset final.csv'
    
    if not os.path.exists(file_path):
        st.error(f"❌ Файл '{file_path}' не найден!")
        st.info("💡 Убедитесь, что файл fair_recrutment_dataset final.csv находится в той же папке")
        return None
    
    try:
        df = pd.read_csv(file_path)
        
        # Маппинг колонок
        df.rename(columns={
            'Skill_Score': 'matchScore',
            'Technical_Test_Score': 'testScore',
            'Certifications_Count': 'attemptCount',
            'Hiring_Decision': 'target_success'
        }, inplace=True)
        
        # Добавляем недостающие поля
        if 'responseTime_days' not in df.columns:
            df['responseTime_days'] = np.random.exponential(2.5, len(df)).clip(0.5, 14).round(1)
        
        # Оставляем только нужные колонки
        required_cols = ['matchScore', 'testScore', 'attemptCount', 'responseTime_days', 'target_success']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"❌ В датасете отсутствует колонка: {col}")
                return None
        
        df = df[required_cols].copy()
        df = df.dropna()
        
        # Приводим типы
        df['matchScore'] = pd.to_numeric(df['matchScore'], errors='coerce').fillna(65)
        df['testScore'] = pd.to_numeric(df['testScore'], errors='coerce').fillna(68)
        df['attemptCount'] = pd.to_numeric(df['attemptCount'], errors='coerce').fillna(1).astype(int)
        
        return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки Kaggle датасета: {e}")
        return None

# ============================================
# БОКОВАЯ ПАНЕЛЬ - ВЫБОР ИСТОЧНИКА
# ============================================
st.sidebar.header("📁 Источник данных")
data_source = st.sidebar.radio(
    "Выберите источник:",
    ["🔥 Firebase Firestore (реальные данные)", "📊 Kaggle датасет (fair_recrutment_dataset)"]
)

# ============================================
# ЗАГРУЗКА ДАННЫХ
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
    with st.spinner("🔄 Загрузка Kaggle датасета..."):
        df = load_kaggle_dataset()
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
    
    # Вкладки
    tab1, tab2, tab3 = st.tabs(["📊 Метрики", "🎯 ROC-кривые", "🔍 Кластеры"])
    
    # ===== ВКЛАДКА 1: МЕТРИКИ =====
    with tab1:
        X = df[['matchScore', 'testScore', 'attemptCount', 'responseTime_days']]
        y = df['target_success']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        
        metrics_lr = {
            'Accuracy': accuracy_score(y_test, lr_model.predict(X_test_scaled)),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba_lr),
        }
        metrics_rf = {
            'Accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba_rf),
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression")
            st.metric("Accuracy", f"{metrics_lr['Accuracy']:.3f}")
            st.metric("ROC-AUC", f"{metrics_lr['ROC-AUC']:.3f}")
        with col2:
            st.subheader("Random Forest")
            st.metric("Accuracy", f"{metrics_rf['Accuracy']:.3f}")
            st.metric("ROC-AUC", f"{metrics_rf['ROC-AUC']:.3f}")
        
        st.subheader("📊 Важность признаков")
        importances = rf_model.feature_importances_
        fig, ax = plt.subplots(figsize=(10, 6))
        features = X.columns
        indices = np.argsort(importances)[::-1]
        ax.barh(range(len(importances)), importances[indices], color='skyblue', edgecolor='black')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel('Важность')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    # ===== ВКЛАДКА 2: ROC-КРИВЫЕ =====
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 8))
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
        ax.plot(fpr_lr, tpr_lr, label=f'LogReg (AUC={metrics_lr["ROC-AUC"]:.3f})', linewidth=2)
        ax.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={metrics_rf["ROC-AUC"]:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-кривые')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # ===== ВКЛАДКА 3: КЛАСТЕРЫ =====
    with tab3:
        scaler_cluster = StandardScaler()
        X_scaled = scaler_cluster.fit_transform(X)
        k = st.slider("Количество кластеров", 2, 6, 4)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(df['matchScore'], df['testScore'],
                                c=df['cluster'], cmap='viridis', alpha=0.6, s=50)
            ax.set_xlabel('matchScore')
            ax.set_ylabel('testScore')
            ax.set_title('Кластеры кандидатов')
            plt.colorbar(scatter, ax=ax, label='Кластер')
            st.pyplot(fig)
        with col2:
            st.write("**Распределение по кластерам:**")
            st.dataframe(df.groupby('cluster').agg({
                'matchScore': 'mean', 'testScore': 'mean',
                'attemptCount': 'mean', 'target_success': 'mean'
            }).round(2))

else:
    st.warning("⚠️ Нет данных для анализа. Выберите источник данных в боковой панели.")

st.markdown("---")
st.markdown("*HR Test Pro Analytics Dashboard v1.0*")
