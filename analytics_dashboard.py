import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from math import pi
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*60)
print("HR Test Pro - ПОЛНЫЙ Аналитический модуль")
print("="*60)

# ============================================
# ЧАСТЬ 1. ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ
# ============================================

print("\n📁 ЧАСТЬ 1: Загрузка данных")

try:
    from google.colab import files
    print("📤 Загрузите файл applications.csv")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    df_real = pd.read_csv(file_name)
    print(f"✅ Загружено {len(df_real)} реальных записей")
except:
    print("⚠️ Файл не загружен или вы не в Colab. Использую тестовые данные.")
    # Создаём минимальные тестовые данные для демонстрации
    df_real = pd.DataFrame({
        'matchScore': [65, 70, 55, 80, 45],
        'status': ['offered', 'offered', 'rejected', 'offered', 'rejected']
    })
    print(f"✅ Создано {len(df_real)} тестовых записей")

# ============================================
# ЧАСТЬ 2. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКИХ ДАННЫХ (n=2000)
# ============================================

print("\n" + "="*60)
print("🔧 ЧАСТЬ 2: Генерация синтетических данных (n=2000)")
print("="*60)

np.random.seed(42)
n_synthetic = 2000

synthetic_data = {
    'matchScore': np.random.normal(65, 15, n_synthetic).clip(20, 98).round(1),
    'testScore': np.random.normal(68, 18, n_synthetic).clip(30, 100).round(1),
    'attemptCount': np.random.choice([1, 2, 3], n_synthetic, p=[0.65, 0.25, 0.10]),
    'responseTime_days': np.random.exponential(2.5, n_synthetic).clip(0.5, 14).round(1),
}

df_synthetic = pd.DataFrame(synthetic_data)

# Генерируем target_success на основе логистической функции
prob_success = 1 / (1 + np.exp(-(
    0.08 * (df_synthetic['matchScore'] - 60) +
    0.05 * (df_synthetic['testScore'] - 65) -
    0.3 * (df_synthetic['attemptCount'] - 1) -
    0.1 * (df_synthetic['responseTime_days'] - 3)
)))
df_synthetic['target_success'] = (np.random.random(n_synthetic) < prob_success).astype(int)

print(f"✅ Сгенерировано {n_synthetic} синтетических записей")
print(f"📊 Распределение целевой переменной:")
print(df_synthetic['target_success'].value_counts())

# ============================================
# ЧАСТЬ 3. ОБЪЕДИНЕНИЕ ДАННЫХ
# ============================================

print("\n" + "="*60)
print("🔗 ЧАСТЬ 3: Объединение данных")
print("="*60)

# Подготовка реальных данных
if 'appliedAt' in df_real.columns:
    df_real_clean = df_real[['matchScore', 'status', 'appliedAt']].copy()
else:
    df_real_clean = df_real[['matchScore', 'status']].copy()
    df_real_clean['appliedAt'] = None

df_real_clean['matchScore'] = pd.to_numeric(df_real_clean['matchScore'], errors='coerce').fillna(50)
df_real_clean['target_success'] = df_real_clean['status'].apply(
    lambda x: 1 if str(x).lower() in ['offered', 'hired', 'accepted'] else 0
)

# Добавляем недостающие колонки для реальных данных
for col in ['testScore', 'attemptCount', 'responseTime_days']:
    if col not in df_real_clean.columns:
        if col == 'testScore':
            df_real_clean[col] = df_real_clean['matchScore'] + np.random.normal(3, 5, len(df_real_clean))
        elif col == 'attemptCount':
            df_real_clean[col] = np.random.choice([1, 2, 3], len(df_real_clean), p=[0.65, 0.25, 0.10])
        else:
            df_real_clean[col] = np.random.exponential(2.5, len(df_real_clean)).clip(0.5, 14)

df_real_clean = df_real_clean[['matchScore', 'testScore', 'attemptCount', 'responseTime_days', 'target_success']].copy()

if df_real_clean is not None and df_synthetic is not None:
    df_combined = pd.concat([df_synthetic, df_real_clean], ignore_index=True)
    print(f"📊 Объединенный датасет: {len(df_combined):,} записей")
    print(f"   - Из синтетических: {len(df_synthetic):,}")
    print(f"   - Из ваших данных: {len(df_real_clean)}")
elif df_synthetic is not None:
    df_combined = df_synthetic.copy()
    print("📊 Использую только синтетические данные")
else:
    print("❌ Нет данных для анализа!")
    df_combined = pd.DataFrame({
        'matchScore': np.random.normal(65, 15, 2000).clip(20, 98).round(1),
        'testScore': np.random.normal(68, 18, 2000).clip(30, 100).round(1),
        'attemptCount': np.random.choice([1, 2, 3], 2000, p=[0.65, 0.25, 0.10]),
        'responseTime_days': np.random.exponential(2.5, 2000).clip(0.5, 14).round(1),
    })
    prob = 1 / (1 + np.exp(-(0.08*(df_combined['matchScore']-60) + 0.05*(df_combined['testScore']-65))))
    df_combined['target_success'] = (np.random.random(2000) < prob).astype(int)

# ============================================
# ЧАСТЬ 4. ПОДГОТОВКА ДАННЫХ ДЛЯ ML
# ============================================

print("\n" + "="*60)
print("🔬 ЧАСТЬ 4: Подготовка данных для машинного обучения")
print("="*60)

X = df_combined[['matchScore', 'testScore', 'attemptCount', 'responseTime_days']]
y = df_combined['target_success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Обучающая выборка: {len(X_train)} записей")
print(f"✅ Тестовая выборка: {len(X_test)} записей")
print(f"📊 Баланс классов в train: {y_train.value_counts().to_dict()}")

# ============================================
# ЧАСТЬ 5. ОБУЧЕНИЕ МОДЕЛЕЙ
# ============================================

print("\n" + "="*60)
print("🤖 ЧАСТЬ 5: Обучение моделей машинного обучения")
print("="*60)

# Логистическая регрессия
print("\n📈 Обучение логистической регрессии...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Random Forest с оптимизацией
print("🌲 Обучение Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("✅ Модели обучены успешно!")

# ============================================
# ЧАСТЬ 6. ОЦЕНКА МОДЕЛЕЙ
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 6: Оценка качества моделей")
print("="*60)

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

results_df = pd.DataFrame([metrics_lr, metrics_rf],
                         index=['Logistic Regression', 'Random Forest (optimized)'])
print("\n📊 Сравнение моделей:")
print(results_df.round(3))

# ============================================
# ЧАСТЬ 7. ROC-КРИВЫЕ
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 7: ROC-кривые")
print("="*60)

plt.figure(figsize=(10, 8))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {metrics_lr["ROC-AUC"]:.3f})', linewidth=2)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {metrics_rf["ROC-AUC"]:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate (1 - Специфичность)')
plt.ylabel('True Positive Rate (Чувствительность)')
plt.title('ROC-кривые для моделей классификации')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ЧАСТЬ 8. МАТРИЦЫ ОШИБОК
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 8: Матрицы ошибок")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Logistic Regression\nAccuracy: {metrics_lr["Accuracy"]:.3f}')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Random Forest\nAccuracy: {metrics_rf["Accuracy"]:.3f}')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ЧАСТЬ 9. ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 9: Важность признаков")
print("="*60)

importances = rf_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Важность признаков (Random Forest)')
plt.bar(range(len(importances)), importances[indices], align='center', color='skyblue', edgecolor='black')
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Признак')
plt.ylabel('Важность')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 Ранжирование признаков по важности:")
for i in range(len(importances)):
    print(f"   {features[indices[i]]}: {importances[indices[i]]:.4f}")

# ============================================
# ЧАСТЬ 10. КЛАСТЕРИЗАЦИЯ КАНДИДАТОВ
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 10: Кластеризация кандидатов")
print("="*60)

scaler_cluster = StandardScaler()
X_scaled_cluster = scaler_cluster.fit_transform(X)

inertia = []
K_range = range(2, 7)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled_cluster)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.title('Метод локтя для определения оптимального k')
plt.grid(True, alpha=0.3)
plt.savefig('elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_combined['cluster'] = kmeans_final.fit_predict(X_scaled_cluster)

print(f"\n✅ Выполнена кластеризация на {optimal_k} кластера")
print("\n📊 Распределение по кластерам:")
print(df_combined['cluster'].value_counts().sort_index())

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, cluster_id in enumerate(range(optimal_k)):
    cluster_data = df_combined[df_combined['cluster'] == cluster_id]
    axes[idx].hist(cluster_data['matchScore'], bins=20, alpha=0.7, edgecolor='black')
    axes[idx].axvline(cluster_data['matchScore'].mean(), color='red', linestyle='--', linewidth=2)
    axes[idx].set_title(f'Кластер {cluster_id}\n(n={len(cluster_data)})')
    axes[idx].set_xlabel('matchScore')
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_combined['matchScore'], df_combined['testScore'],
                     c=df_combined['cluster'], cmap='viridis', alpha=0.6, s=50)
plt.xlabel('matchScore')
plt.ylabel('testScore')
plt.title('Кластеры кандидатов')
plt.colorbar(scatter, label='Кластер')
plt.tight_layout()
plt.savefig('clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ЧАСТЬ 11. АНАЛИЗ СЕЗОННОСТИ
# ============================================

print("\n" + "="*60)
print("📅 ЧАСТЬ 11: Анализ сезонности откликов")
print("="*60)

all_dates = []

if 'appliedAt' in df_real.columns:
    real_dates = df_real['appliedAt'].dropna()
    real_dates_parsed = pd.to_datetime(real_dates, errors='coerce')
    all_dates.extend(real_dates_parsed.dropna().tolist())
    print(f"📌 Реальных дат: {len(real_dates_parsed.dropna())}")

if len(all_dates) < 100:
    print("⚠️ Реальных дат мало. Добавляем синтетические...")
    base_date = pd.Timestamp('2024-01-01')
    synthetic_dates = [base_date + pd.Timedelta(days=np.random.randint(0, 365)) for _ in range(500)]
    all_dates.extend(synthetic_dates)
    print(f"📌 Добавлено 500 синтетических дат")

df_all_dates = pd.DataFrame({'appliedAt': all_dates})
df_all_dates['month'] = df_all_dates['appliedAt'].dt.month
df_all_dates['day_of_week'] = df_all_dates['appliedAt'].dt.dayofweek
df_all_dates['hour'] = df_all_dates['appliedAt'].dt.hour

print(f"\n📊 Всего дат для анализа: {len(df_all_dates)}")

monthly_counts = df_all_dates.groupby('month').size()
months_ru = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
             'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].bar(months_ru, monthly_counts.values, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Месяц')
axes[0, 0].set_ylabel('Количество откликов')
axes[0, 0].set_title('Распределение откликов по месяцам')
axes[0, 0].tick_params(axis='x', rotation=45)

seasonal_coef = monthly_counts / monthly_counts.mean()
colors_coef = ['red' if x < 0.9 else 'green' if x > 1.1 else 'gray' for x in seasonal_coef.values]
axes[0, 1].bar(months_ru, seasonal_coef.values, color=colors_coef, edgecolor='black')
axes[0, 1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7)
axes[0, 1].set_xlabel('Месяц')
axes[0, 1].set_ylabel('Сезонный коэффициент')
axes[0, 1].set_title('Сезонные коэффициенты')
axes[0, 1].tick_params(axis='x', rotation=45)

weekday_counts = df_all_dates.groupby('day_of_week').size().reindex(range(0, 7), fill_value=0)
days_ru = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
axes[1, 0].bar(days_ru, weekday_counts.values, color='coral', edgecolor='black')
axes[1, 0].set_xlabel('День недели')
axes[1, 0].set_ylabel('Количество откликов')
axes[1, 0].set_title('Распределение по дням недели')

if 'hour' in df_all_dates.columns:
    hourly_counts = df_all_dates.groupby('hour').size().reindex(range(0, 24), fill_value=0)
    axes[1, 1].plot(hourly_counts.index, hourly_counts.values, 'o-', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Час суток')
    axes[1, 1].set_ylabel('Количество откликов')
    axes[1, 1].set_title('Распределение по часам')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('seasonality_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n📊 СЕЗОННЫЕ КОЭФФИЦИЕНТЫ:")
for i, (month, coef) in enumerate(seasonal_coef.items()):
    trend = "🔺 ВЫСОКИЙ" if coef > 1.1 else "🔻 НИЗКИЙ" if coef < 0.9 else "➖ СРЕДНИЙ"
    print(f"   {months_ru[i]}: {coef:.2f} - {trend}")

# ============================================
# ЧАСТЬ 12. АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ
# ============================================

print("\n" + "="*60)
print("📊 ЧАСТЬ 12: Анализ чувствительности модели")
print("="*60)

thresholds = np.arange(0.1, 0.9, 0.05)
precision_scores = []
recall_scores = []
f1_scores = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba_rf >= thresh).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_thresh, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, 'b-o', label='Precision', linewidth=2)
plt.plot(thresholds, recall_scores, 'r-o', label='Recall', linewidth=2)
plt.plot(thresholds, f1_scores, 'g-o', label='F1-Score', linewidth=2)
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Порог по умолчанию (0.5)')
plt.xlabel('Порог классификации')
plt.ylabel('Значение метрики')
plt.title('Анализ чувствительности к порогу классификации')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_idx = np.argmax(f1_scores)
print(f"\n✅ Оптимальный порог: {thresholds[optimal_idx]:.2f}")
print(f"   F1-Score: {f1_scores[optimal_idx]:.3f}")
print(f"   Precision: {precision_scores[optimal_idx]:.3f}")
print(f"   Recall: {recall_scores[optimal_idx]:.3f}")

# ============================================
# ЧАСТЬ 13. ПРОФИЛЬ ИДЕАЛЬНОГО КАНДИДАТА
# ============================================

print("\n" + "="*60)
print("👑 ЧАСТЬ 13: Профиль идеального кандидата")
print("="*60)

successful_candidates = df_combined[df_combined['target_success'] == 1]

ideal_profile = {
    'matchScore': successful_candidates['matchScore'].mean(),
    'testScore': successful_candidates['testScore'].mean(),
    'attemptCount': successful_candidates['attemptCount'].mode().iloc[0] if len(successful_candidates['attemptCount'].mode()) > 0 else 1,
    'responseTime_days': successful_candidates['responseTime_days'].median()
}

print("\n📊 ХАРАКТЕРИСТИКИ УСПЕШНЫХ КАНДИДАТОВ:")
print(f"   matchScore: {ideal_profile['matchScore']:.1f} ± {successful_candidates['matchScore'].std():.1f}")
print(f"   testScore: {ideal_profile['testScore']:.1f} ± {successful_candidates['testScore'].std():.1f}")
print(f"   attemptCount: {ideal_profile['attemptCount']:.1f} (мода)")
print(f"   responseTime_days: {ideal_profile['responseTime_days']:.1f} дней (медиана)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist([df_combined['matchScore'], successful_candidates['matchScore']],
               bins=20, label=['Все кандидаты', 'Успешные'], alpha=0.7, color=['gray', 'green'])
axes[0, 0].axvline(ideal_profile['matchScore'], color='green', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('matchScore')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Распределение matchScore')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist([df_combined['testScore'], successful_candidates['testScore']],
               bins=20, label=['Все кандидаты', 'Успешные'], alpha=0.7, color=['gray', 'blue'])
axes[0, 1].axvline(ideal_profile['testScore'], color='blue', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('testScore')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Распределение testScore')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist([df_combined['attemptCount'], successful_candidates['attemptCount']],
               bins=[0.5, 1.5, 2.5, 3.5], label=['Все кандидаты', 'Успешные'], alpha=0.7, color=['gray', 'orange'])
axes[1, 0].set_xlabel('attemptCount')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Распределение attemptCount')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist([df_combined['responseTime_days'], successful_candidates['responseTime_days']],
               bins=20, label=['Все кандидаты', 'Успешные'], alpha=0.7, color=['gray', 'red'])
axes[1, 1].axvline(ideal_profile['responseTime_days'], color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('responseTime_days')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Распределение responseTime_days')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_candidate_profile.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ЧАСТЬ 14. ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ
# ============================================

print("\n" + "="*60)
print("📈 ЧАСТЬ 14: Прогнозирование временных рядов")
print("="*60)

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    
    if len(df_all_dates) > 100:
        daily_counts = df_all_dates.groupby(df_all_dates['appliedAt'].dt.date).size()
        daily_counts = daily_counts.reindex(pd.date_range(start=daily_counts.index.min(),
                                                          end=daily_counts.index.max(),
                                                          freq='D'), fill_value=0)
        
        print(f"\n📊 Анализ временного ряда: {len(daily_counts)} дней")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(daily_counts.index, daily_counts.values, linewidth=1)
        axes[0].set_title('Ежедневное количество откликов')
        axes[0].set_xlabel('Дата')
        axes[0].set_ylabel('Количество')
        axes[0].grid(True, alpha=0.3)
        
        try:
            decomposition = seasonal_decompose(daily_counts, model='additive', period=7)
            
            axes[1].plot(decomposition.trend, label='Trend', linewidth=2)
            axes[1].plot(decomposition.seasonal, label='Seasonal', linewidth=1, alpha=0.7)
            axes[1].set_title('Декомпозиция временного ряда')
            axes[1].set_xlabel('Дата')
            axes[1].set_ylabel('Значение')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('time_series_decomposition.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            model = ARIMA(daily_counts, order=(1, 1, 1))
            model_fit = model.fit()
            
            forecast_steps = 30
            forecast = model_fit.forecast(steps=forecast_steps)
            
            plt.figure(figsize=(12, 6))
            plt.plot(daily_counts.index, daily_counts.values, label='Historical', linewidth=1)
            plt.plot(forecast.index, forecast.values, label='Forecast', linewidth=2, color='red')
            plt.fill_between(forecast.index,
                           forecast.values - forecast.values.std(),
                           forecast.values + forecast.values.std(),
                           alpha=0.3, color='red')
            plt.title(f'Прогноз количества откликов на {forecast_steps} дней')
            plt.xlabel('Дата')
            plt.ylabel('Количество откликов')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('arima_forecast.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\n✅ Прогноз на {forecast_steps} дней выполнен")
            print(f"   Среднее прогнозируемое значение: {forecast.mean():.1f} откликов/день")
            
        except Exception as e:
            print(f"⚠️ Ошибка при декомпозиции: {e}")
    else:
        print("⚠️ Недостаточно данных для анализа временных рядов")
        
except ImportError:
    print("⚠️ Установите statsmodels для анализа временных рядов: pip install statsmodels")
except Exception as e:
    print(f"⚠️ Ошибка при прогнозировании: {e}")

# ============================================
# ЧАСТЬ 15. ИТОГОВЫЕ ВЫВОДЫ
# ============================================

print("\n" + "="*60)
print("📝 ЧАСТЬ 15: Итоговые выводы по ML-аналитике")
print("="*60)

print(f"""
🎯 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:

1. МЕТРИКИ МОДЕЛЕЙ:
   - Логистическая регрессия: Accuracy = {metrics_lr['Accuracy']:.3f}, AUC = {metrics_lr['ROC-AUC']:.3f}
   - Random Forest: Accuracy = {metrics_rf['Accuracy']:.3f}, AUC = {metrics_rf['ROC-AUC']:.3f}

2. ВАЖНЕЙШИЕ ПРИЗНАКИ:
   - {features[indices[0]]}: {importances[indices[0]]:.4f}
   - {features[indices[1]]}: {importances[indices[1]]:.4f}
   - {features[indices[2]]}: {importances[indices[2]]:.4f}

3. ПРОФИЛЬ УСПЕШНОГО КАНДИДАТА:
   - matchScore: {ideal_profile['matchScore']:.1f}
   - testScore: {ideal_profile['testScore']:.1f}
   - attemptCount: {ideal_profile['attemptCount']:.0f}
   - responseTime: {ideal_profile['responseTime_days']:.1f} дней

4. СЕЗОННОСТЬ:
   - Пиковые месяцы: {', '.join([months_ru[i] for i, x in enumerate(seasonal_coef.values) if x > 1.1])}
   - Низкие месяцы: {', '.join([months_ru[i] for i, x in enumerate(seasonal_coef.values) if x < 0.9])}

5. КЛАСТЕРИЗАЦИЯ:
   - Выделено {optimal_k} кластеров кандидатов
   - Распределение: {df_combined['cluster'].value_counts().sort_index().to_dict()}

📊 СОХРАНЕННЫЕ ФАЙЛЫ:
   - roc_curves.png - ROC-кривые
   - confusion_matrices.png - Матрицы ошибок
   - feature_importance.png - Важность признаков
   - cluster_distributions.png - Распределение кластеров
   - clusters.png - Визуализация кластеров
   - seasonality_analysis.png - Анализ сезонности
   - threshold_analysis.png - Анализ чувствительности
   - ideal_candidate_profile.png - Профиль идеального кандидата
   - time_series_decomposition.png - Декомпозиция временного ряда
   - arima_forecast.png - Прогноз ARIMA

 РЕКОМЕНДАЦИИ:
   1. Сфокусируйтесь на кандидатах с matchScore > {ideal_profile['matchScore']:.0f}
   2. Оптимальный порог классификации: {thresholds[optimal_idx]:.2f}
   3. Учитывайте сезонность при планировании рекрутинга
   4. Улучшайте процесс для 1-й попытки (attemptCount=1)

""")

print("="*60)
print("✅ АНАЛИТИЧЕСКИЙ МОДУЛЬ УСПЕШНО ВЫПОЛНЕН")
print("="*60)
