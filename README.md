# HR Test Pro - Аналитический модуль

📊 **Интерактивный дашборд для глубокой аналитики подбора персонала**

## Описание

Веб-приложение на Streamlit для расширенной аналитики результатов тестирования кандидатов. Интегрируется с мобильным приложением HR Test Pro через Firebase Firestore.

## Функциональность

- 📈 **Визуализация данных**: распределение баллов, метрики эффективности
- 🎯 **ML-прогнозирование**: прогноз успешности кандидатов (Random Forest, AUC-ROC 0.84)
- 🔍 **Кластеризация**: сегментация кандидатов по компетенциям (K-means)
- 📅 **Анализ сезонности**: выявление пиков откликов и трендов
- 💡 **Рекомендации**: автоматические рекомендации для HR на основе данных
- 📄 **Экспорт**: формирование брендированных PDF-отчётов

## Технологии

- **Frontend**: Streamlit 1.31.0
- **Data Processing**: Pandas, NumPy
- **ML**: Scikit-learn (Random Forest, K-means)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Backend**: Firebase Admin SDK
- **Database**: Cloud Firestore

## Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/ВАШ_USERNAME/hr-test-pro-analytics.git
cd streamlit-analytics