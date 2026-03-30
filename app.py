import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка страницы
st.set_page_config(page_title="AI Customer Segmentation", layout="wide")

st.title("🤖 Интеллектуальная система сегментации клиентов")
st.write("Это приложение использует машинное обучение для автоматического определения профилей ваших клиентов.")

# 1. Загрузка файла в боковой панели
st.sidebar.header("📁 Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Файл успешно загружен!")
    
    # 2. Настройки ИИ
    st.sidebar.header("⚙️ Настройки алгоритма")
    k = st.sidebar.slider("Количество сегментов (K)", 2, 10, 5)
    
    # Подготовка признаков (берем доход и рейтинг трат)
    features = ['Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features]
    
    # Масштабирование данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Обучение модели KMeans
    model = KMeans(n_clusters=k, init='k-means++', random_state=42)
    df['Cluster'] = model.fit_predict(X_scaled)
    
    # 3. Визуализация и Анализ
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📍 Карта сегментов")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            data=df, x=features[0], y=features[1], 
            hue='Cluster', palette='viridis', s=150, ax=ax, alpha=0.7
        )
        plt.title(f"Разделение на {k} групп")
        st.pyplot(fig)
        
    with col2:
        st.subheader("📝 Автоматическая интерпретация")
        analysis = df.groupby('Cluster')[features].mean()
        
        for i in range(k):
            income = analysis.loc[i, features[0]]
            spending = analysis.loc[i, features[1]]
            
            # Логика классификации
            if income > 70 and spending > 70:
                label = "💎 VIP-клиенты"
                color = "green"
                advice = "Фокус на удержании, эксклюзивные предложения."
            elif income > 70 and spending < 40:
                label = "💰 Экономные богачи"
                color = "blue"
                advice = "Нужны скидки за объем и долгосрочные выгоды."
            elif income < 45 and spending > 70:
                label = "🛍️ Активные транжиры"
                color = "orange"
                advice = "Идеально для новинок и импульсивных покупок."
            elif income < 45 and spending < 40:
                label = "🐌 Малоактивные"
                color = "gray"
                advice = "Требуются агрессивные акции для активации."
            else:
                label = "📊 Средний класс"
                color = "black"
                advice = "Стандартная поддержка лояльности."

            # Вывод карточки сегмента
            with st.expander(f"Сегмент {i}: {label}"):
                st.write(f"**Доход:** {income:.1f}k$ | **Траты:** {spending:.1f}")
                st.info(f"💡 {advice}")

    # 4. Просмотр данных
    st.divider()
    st.subheader("📂 Итоговая база данных с метками ИИ")
    st.dataframe(df.style.background_gradient(subset=['Cluster'], cmap='viridis'))
    
    # Кнопка скачивания результата
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать результат в CSV",
        data=csv,
        file_name='segmented_customers.csv',
        mime='text/csv',
    )

else:
    st.info("Пожалуйста, загрузите файл '......csv' через боковую панель, чтобы запустить анализ.")