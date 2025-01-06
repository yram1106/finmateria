import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de la página
st.set_page_config(
    page_title="Análisis Interactivo de Datos",
    page_icon="📊",
    layout="wide"
)

# Menú de navegación
menu = ["Inicio 🏠", "EDA 📊", "Regresiones 📈", "Informe 📄"]
choice = st.sidebar.selectbox("Navegación", menu)

# Variable para almacenar el dataset cargado
if "dataset" not in st.session_state:
    st.session_state.dataset = None

# Inicio
if choice == "Inicio 🏠":
    st.title("Bienvenido a la Aplicación de Análisis Interactivo de Datos")
    st.write("Usa el menú de la izquierda para navegar entre los módulos.")
    st.write("Puedes cargar tus datos, explorar estadísticas, aplicar regresiones y generar informes.")

# EDA
elif choice == "EDA 📊":
    st.title("Análisis Exploratorio de Datos 📊")
    st.write("Carga un dataset y realiza análisis exploratorios.")

    # Cargar dataset
    uploaded_file = st.file_uploader("Carga tu archivo CSV", type=["csv"])
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.session_state.dataset = dataset
        st.write("Dataset cargado exitosamente:")
        st.dataframe(dataset.head())
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        st.write(dataset.describe())

        # Visualizaciones
        st.subheader("Visualizaciones")
        col_x = st.selectbox("Selecciona la variable para el eje X", dataset.columns)
        col_y = st.selectbox("Selecciona la variable para el eje Y", dataset.columns)
        plot_type = st.radio("Selecciona el tipo de gráfico", ["Scatterplot", "Boxplot"])

        if plot_type == "Scatterplot":
            fig, ax = plt.subplots()
            ax.scatter(dataset[col_x], dataset[col_y])
            ax.set_title(f"Scatterplot: {col_x} vs {col_y}")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            st.pyplot(fig)
        elif plot_type == "Boxplot":
            fig, ax = plt.subplots()
            ax.boxplot(dataset[col_y])
            ax.set_title(f"Boxplot de {col_y}")
            ax.set_ylabel(col_y)
            st.pyplot(fig)

# Regresiones
elif choice == "Regresiones 📈":
    st.title("Módulo de Regresiones 📈")
    st.write("Aplica modelos de regresión a tus datos.")

    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        st.write("Datos cargados:")
        st.dataframe(dataset.head())

        # Seleccionar variables
        st.subheader("Configuración del modelo")
        target = st.selectbox("Selecciona la variable objetivo (Y)", dataset.columns)
        features = st.multiselect("Selecciona las variables predictoras (X)", dataset.columns)

        if st.button("Entrenar modelo"):
            X = dataset[features]
            y = dataset[target]

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Entrenar modelo
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predicciones y métricas
            y_pred = model.predict(X_test)
            st.write("Coeficientes del modelo:", model.coef_)
            st.write("Intercepto:", model.intercept_)
            st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
            st.write("R² Score:", r2_score(y_test, y_pred))

            # Visualizar regresión
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Valores Reales")
            ax.set_ylabel("Predicciones")
            st.pyplot(fig)
    else:
        st.warning("Por favor, carga un dataset en la sección de EDA primero.")

# Informe
elif choice == "Informe 📄":
    st.title("Generación de Informes 📄")
    st.write("Crea y descarga informes ejecutivos de tu análisis.")

    if st.session_state.dataset is not None:
        dataset = st.session_state.dataset
        st.write("Generando informe...")

        report_html = f"""
        <h1>Informe Ejecutivo</h1>
        <p>Este informe contiene los resultados del análisis exploratorio de datos y los modelos de regresión aplicados.</p>
        <h2>Vista Previa de los Datos</h2>
        {dataset.head().to_html()}
        <h2>Estadísticas Descriptivas</h2>
        {dataset.describe().to_html()}
        """

        # Mostrar informe como HTML si no se puede generar PDF
        st.markdown(report_html, unsafe_allow_html=True)
        
        # Alternativa si pdfkit no funciona: permitir descargar un CSV
        if st.button("Descargar datos como CSV"):
            dataset.to_csv("informe.csv")
            with open("informe.csv", "rb") as file:
                st.download_button(
                    label="Descargar Informe CSV",
                    data=file,
                    file_name="informe.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Por favor, carga un dataset en la sección de EDA primero.")

# Footer
st.sidebar.write("Desarrollado por Mary Figueroa - Paradigmas de programación")
