import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

st.set_page_config(page_title="HR Test Pro - Аналитика", layout="wide")
st.title("📊 HR Test Pro - Глубокая аналитика")

# ============================================
# ЗАГРУЗКА ДАННЫХ
# ============================================
st.sidebar.header("📁 Источник данных")
data_source = st.sidebar.radio("Выберите:", ["Демо-данные (2000 записей)", "Загрузить CSV"])

@st.cache_data
def generate_synthetic_data(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        'matchScore': np.random.normal(65, 15, n).clip(20, 98).round(1),
        'testScore': np.random.normal(68, 18, n).clip(30, 100).round(1),
        'attemptCount': np.random.choice([1, 2, 3], n, p=[0.65, 0.25, 0.10]),
        'responseTime_days': np.random.exponential(2.5, n).clip(0.5, 14).round(1),
    })
    prob = 1 / (1 + np.exp(-(0.08*(df['matchScore']-60) + 0.05*(df['testScore']-65) - 0.3*(df['attemptCount']-1) - 0.1*(df['responseTime_days']-3))))
    df['target_success'] = (np.random.random(n) < prob).astype(int)
    return df

if data_source == "Загрузить CSV":
    uploaded = st.sidebar.file_uploader("CSV файл", type=['csv'])
    if uploaded:
        df_combined = pd.read_csv(uploaded)
        st.sidebar.success(f"✅ Загружено {len(df_combined)} записей")
    else:
        st.info("Загрузите файл для анализа")
        st.stop()
else:
    df_combined = generate_synthetic_data()
    st.sidebar.success(f"✅ Сгенерировано {len(df_combined)} записей")

# ============================================
# ML МОДЕЛИ
# ============================================
X = df_combined[['matchScore', 'testScore', 'attemptCount', 'responseTime_days']]
y = df_combined['target_success']
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

# Метрики
metrics_lr = {
    'Accuracy': accuracy_score(y_test, lr_model.predict(X_test_scaled)),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_lr),
}
metrics_rf = {
    'Accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_rf),
}

# ============================================
# ВКЛАДКИ ДАШБОРДА
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Метрики", "🎯 ROC-кривые", "🔍 Кластеры", "📅 Сезонность", "👑 Профиль"])

with tab1:
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

with tab3:
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    k = st.slider("Количество кластеров", 2, 6, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_combined['cluster'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(df_combined['matchScore'], df_combined['testScore'],
                            c=df_combined['cluster'], cmap='viridis', alpha=0.6, s=50)
        ax.set_xlabel('matchScore')
        ax.set_ylabel('testScore')
        ax.set_title('Кластеры кандидатов')
        plt.colorbar(scatter, ax=ax, label='Кластер')
        st.pyplot(fig)
    with col2:
        st.write("**Распределение по кластерам:**")
        st.dataframe(df_combined.groupby('cluster').agg({
            'matchScore': 'mean', 'testScore': 'mean',
            'attemptCount': 'mean', 'target_success': 'mean'
        }).round(2))

with tab4:
    st.info("📅 Анализ сезонности доступен при загрузке реальных данных с датами")
    # Здесь можно добавить анализ, если в данных есть колонка 'appliedAt'

with tab5:
    successful = df_combined[df_combined['target_success'] == 1]
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Средний matchScore (успешные)", f"{successful['matchScore'].mean():.1f}")
        st.metric("Средний testScore (успешные)", f"{successful['testScore'].mean():.1f}")
    with col2:
        st.metric("Мода attemptCount", f"{successful['attemptCount'].mode().iloc[0]:.0f}")
        st.metric("Медиана responseTime", f"{successful['responseTime_days'].median():.1f} дн.")
    
    st.success(f"💡 Рекомендация: фокус на кандидатах с matchScore > {successful['matchScore'].quantile(0.25):.0f}")
