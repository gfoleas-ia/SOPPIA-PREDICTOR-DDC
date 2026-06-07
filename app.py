
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

DATASET = "dataset_sintetico_ddc.csv"
ICONO = "assets/soppia_sofia_icon.png"
DICCIONARIO = "diccionario_variables_ddc.txt"
OBJETIVO = "ddc_diagnostico"

try:
    icono_sofia = Image.open(ICONO)
except Exception:
    icono_sofia = "🤖"

st.set_page_config(
    page_title="SOPP+IA Sofía | Predictor DDC",
    page_icon=icono_sofia,
    layout="wide"
)

# ============================================================
# ESTILO VISUAL
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #004A98, #003B7A);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    .titulo {
        color: #003B7A;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 10px;
    }

    .subtitulo {
        color: #334155;
        font-size: 18px;
        line-height: 1.5;
    }

    .alerta {
        background-color: #E8F2FF;
        color: #003B7A;
        padding: 15px;
        border-radius: 10px;
        border-left: 7px solid #F26A21;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .card {
        background-color: white;
        border: 1px solid #D6E4F5;
        border-radius: 15px;
        padding: 18px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0, 59, 122, 0.08);
        min-height: 95px;
    }

    .numero {
        color: #003B7A;
        font-size: 28px;
        font-weight: 800;
    }

    .texto {
        color: #475569;
        font-size: 15px;
    }

    .bloque {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .bloque-azul {
        background-color: #F0F7FF;
        border: 1px solid #CFE3FF;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .bibliografia {
        background-color: #FFFFFF;
        border-left: 5px solid #003B7A;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 15px;
        color: #334155;
    }

    .footer {
        margin-top: 35px;
        padding: 16px;
        background-color: #E8F2FF;
        border-radius: 10px;
        text-align: center;
        color: #003B7A;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# FUNCIONES DE DATOS Y MODELO
# ============================================================

@st.cache_data
def cargar_datos():
    df = pd.read_csv(DATASET)

    if OBJETIVO in df.columns:
        df[OBJETIVO] = (
            df[OBJETIVO]
            .astype(str)
            .str.strip()
            .replace({
                "Sí": "Si",
                "sí": "Si",
                "SI": "Si",
                "si": "Si",
                "No ": "No",
                "NO": "No",
                "no": "No"
            })
        )

    return df


@st.cache_resource
def entrenar_modelo(df):
    datos = df.copy()

    columnas_excluir = [
        "id_paciente",
        "riesgo_teorico_ddc"
    ]

    datos = datos.drop(columns=columnas_excluir, errors="ignore")

    X = datos.drop(columns=[OBJETIVO])
    y = datos[OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    columnas_categoricas = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    columnas_numericas = X.select_dtypes(
        exclude=["object", "category", "bool"]
    ).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", "passthrough", columnas_numericas)
        ]
    )

    modelo_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo_rf)
        ]
    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    metricas = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Si", zero_division=0),
        "Recall": recall_score(y_test, y_pred, pos_label="Si", zero_division=0),
        "F1-score": f1_score(y_test, y_pred, pos_label="Si", zero_division=0)
    }

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=["No", "Si"]
    )

    return modelo, X, metricas, matriz


def obtener_probabilidad_ddc(modelo, nuevo_paciente):
    clases = list(modelo.named_steps["modelo"].classes_)
    probabilidades = modelo.predict_proba(nuevo_paciente)[0]

    if "Si" in clases:
        indice_si = clases.index("Si")
    else:
        indice_si = 1

    return float(probabilidades[indice_si])


def clasificar_riesgo(probabilidad):
    if probabilidad >= 0.70:
        return (
            "Riesgo alto",
            "Evaluación prioritaria por Ortopedia Pediátrica."
        )

    elif probabilidad >= 0.40:
        return (
            "Riesgo intermedio",
            "Correlacionar con examen físico y estudio de imagen."
        )

    else:
        return (
            "Riesgo bajo",
            "Mantener vigilancia clínica según edad, antecedentes y signos físicos."
        )


def card(titulo, valor):
    st.markdown(
        f"""
        <div class="card">
            <div class="numero">{valor}</div>
            <div class="texto">{titulo}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def mostrar_bibliografia():
    referencias = [
        "Agostiniani, R., et al. (2020). Recommendations for early diagnosis of Developmental Dysplasia of the Hip (DDH): Working group intersociety consensus document. Italian Journal of Pediatrics, 46(150).",
        "Aarvold, A., et al. (2023). The management of developmental dysplasia of the hip in children aged under three months: A consensus study from the British Society for Children’s Orthopaedic Surgery. Bone & Joint Journal, 105-B(2), 209–214.",
        "Kuitunen, I., et al. (2022). Incidence of Neonatal Developmental Dysplasia of the Hip and Late Detection Rates Based on Screening Strategy: A Systematic Review and Meta-analysis. JAMA Network Open, 5(8).",
        "Oleas Santillán, G. F. (2026). Scikit-learn, PyCaret, LazyPredict y Streamlit aplicados a la Ortopedia Pediátrica. Documento académico inédito."
    ]

    for i, ref in enumerate(referencias, start=1):
        st.markdown(
            f"""
            <div class="bibliografia">
            <b>{i}.</b> {ref}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CARGAR DATASET Y ENTRENAR
# ============================================================

try:
    df = cargar_datos()
except Exception:
    st.error("No se encontró el archivo dataset_sintetico_ddc.csv.")
    st.stop()

if OBJETIVO not in df.columns:
    st.error(f"El dataset debe contener la columna objetivo: {OBJETIVO}")
    st.stop()

modelo, X, metricas, matriz = entrenar_modelo(df)

# ============================================================
# MENÚ LATERAL
# ============================================================

try:
    st.sidebar.image(ICONO, width=130)
except Exception:
    st.sidebar.write("🤖")

st.sidebar.title("SOPP+IA Sofía")
st.sidebar.write("Predictor educativo de DDC")

menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Exploración de datos",
        "Modelo predictivo",
        "Predicción individual",
        "Diccionario de variables",
        "Bibliografía",
        "Interpretación clínica"
    ]
)

# ============================================================
# INICIO
# ============================================================

if menu == "Inicio":

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown(
            """
            <div class="titulo">
            SOPP+IA Sofía: Predictor educativo de Displasia del Desarrollo de Cadera
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p class="subtitulo">
            Aplicación académica en Python y Streamlit para explorar factores de riesgo,
            visualizar datos y estimar una probabilidad educativa de Displasia del Desarrollo
            de Cadera mediante Machine Learning.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="alerta">
            Uso académico y educativo. Este sistema no reemplaza el criterio médico,
            el examen físico ni los estudios de imagen.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        try:
            st.image(ICONO, caption="SOPP+IA Sofía", use_container_width=True)
        except Exception:
            st.info("SOPP+IA Sofía")

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Pacientes", df.shape[0])

    with c2:
        card("Variables", df.shape[1] - 1)

    with c3:
        card("Casos DDC", int((df[OBJETIVO] == "Si").sum()))

    with c4:
        card("Modelo", "Random Forest")

    st.write("")

    col_grafico, col_lectura = st.columns([1.4, 1])

    with col_grafico:
        st.subheader("Distribución del diagnóstico")

        fig, ax = plt.subplots(figsize=(6, 4))
        df[OBJETIVO].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Diagnóstico DDC")
        ax.set_ylabel("Número de pacientes")
        ax.set_title("Distribución de la variable objetivo")
        st.pyplot(fig)

    with col_lectura:
        st.subheader("Lectura rápida")

        st.markdown(
            """
            <div class="bloque-azul">
            <b>¿Qué hace Sofía?</b><br>
            Organiza variables clínicas, entrena un modelo predictivo y entrega una
            estimación de riesgo que debe ser interpretada por el médico.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="bloque">
            <b>Uso recomendado:</b><br>
            educación médica, demostración académica, análisis de datos y apoyo
            a la investigación en Ortopedia Pediátrica.
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# EXPLORACIÓN DE DATOS
# ============================================================

elif menu == "Exploración de datos":

    st.header("Exploración del dataset")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Primeras filas")
        st.dataframe(df.head(15), use_container_width=True)

    with col2:
        st.subheader("Resumen general")

        resumen = pd.DataFrame({
            "Indicador": [
                "Número de pacientes",
                "Número de variables",
                "Datos faltantes",
                "Variable objetivo"
            ],
            "Valor": [
                df.shape[0],
                df.shape[1],
                int(df.isnull().sum().sum()),
                OBJETIVO
            ]
        })

        st.table(resumen)

    st.subheader("Visualización por variable")

    variable = st.selectbox(
        "Seleccione una variable",
        [c for c in df.columns if c != "id_paciente"]
    )

    col_grafico, col_texto = st.columns([1.4, 1])

    with col_grafico:
        fig, ax = plt.subplots(figsize=(6, 4))

        if pd.api.types.is_numeric_dtype(df[variable]):
            ax.hist(df[variable].dropna())
            ax.set_xlabel(variable)
            ax.set_ylabel("Frecuencia")
            ax.set_title(f"Histograma de {variable}")
        else:
            df[variable].astype(str).value_counts().plot(kind="bar", ax=ax)
            ax.set_xlabel(variable)
            ax.set_ylabel("Frecuencia")
            ax.set_title(f"Distribución de {variable}")

        st.pyplot(fig)

    with col_texto:
        st.markdown(
            """
            <div class="bloque-azul">
            <b>Interpretación:</b><br>
            Esta sección permite observar cómo se distribuyen las variables del dataset.
            La exploración de datos es el primer paso antes de entrenar un modelo.
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# MODELO PREDICTIVO
# ============================================================

elif menu == "Modelo predictivo":

    st.header("Modelo predictivo")

    st.markdown(
        """
        <div class="bloque-azul">
        El modelo utiliza Random Forest Classifier. Las variables categóricas se transforman
        mediante One-Hot Encoding y las variables numéricas pasan directamente al modelo.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Métricas")

        metricas_df = pd.DataFrame({
            "Métrica": list(metricas.keys()),
            "Valor": [round(v, 3) for v in metricas.values()]
        })

        st.table(metricas_df)

    with col2:
        st.subheader("Matriz de confusión")

        matriz_df = pd.DataFrame(
            matriz,
            index=["Real No", "Real Si"],
            columns=["Predicho No", "Predicho Si"]
        )

        st.table(matriz_df)

    st.warning(
        "Las métricas corresponden a un dataset sintético. No deben interpretarse como validación clínica externa."
    )

# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================

elif menu == "Predicción individual":

    st.header("Predicción individual con SOPP+IA Sofía")

    st.markdown(
        """
        <div class="alerta">
        Ingrese los datos del paciente. La predicción es educativa y requiere interpretación médica profesional.
        </div>
        """,
        unsafe_allow_html=True
    )

    entrada = {}

    with st.form("formulario_prediccion"):

        col1, col2 = st.columns(2)

        for i, columna in enumerate(X.columns):

            contenedor = col1 if i % 2 == 0 else col2

            with contenedor:

                if pd.api.types.is_numeric_dtype(X[columna]):

                    entrada[columna] = st.number_input(
                        label=columna,
                        value=float(X[columna].mean())
                    )

                else:
                    opciones = sorted(
                        X[columna]
                        .astype(str)
                        .dropna()
                        .unique()
                        .tolist()
                    )

                    entrada[columna] = st.selectbox(
                        label=columna,
                        options=opciones
                    )

        boton = st.form_submit_button("Analizar con SOPP+IA Sofía")

    if boton:

        nuevo_paciente = pd.DataFrame([entrada])

        prediccion = modelo.predict(nuevo_paciente)[0]
        probabilidad = obtener_probabilidad_ddc(modelo, nuevo_paciente)
        categoria, recomendacion = clasificar_riesgo(probabilidad)

        st.subheader("Resultado")

        r1, r2, r3 = st.columns(3)

        with r1:
            card("Clasificación", prediccion)

        with r2:
            card("Probabilidad DDC", f"{probabilidad:.1%}")

        with r3:
            card("Categoría", categoria)

        st.write("")

        col_resultado, col_recomendacion = st.columns([1, 1])

        with col_resultado:
            if prediccion == "Si":
                st.warning("El modelo clasifica el caso como compatible con riesgo de DDC.")
            else:
                st.success("El modelo clasifica el caso como bajo riesgo de DDC.")

        with col_recomendacion:
            st.info(recomendacion)

        with st.expander("Ver datos ingresados"):
            st.dataframe(nuevo_paciente, use_container_width=True)

# ============================================================
# DICCIONARIO DE VARIABLES
# ============================================================

elif menu == "Diccionario de variables":

    st.header("Diccionario de variables")

    st.markdown(
        """
        <div class="bloque-azul">
        Esta sección hace público el diccionario de variables utilizado en el dataset.
        Su objetivo es facilitar la lectura clínica, académica y metodológica del predictor.
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        with open(DICCIONARIO, "r", encoding="utf-8") as archivo:
            texto_diccionario = archivo.read()

        lineas = [linea.strip() for linea in texto_diccionario.splitlines() if linea.strip()]

        if len(lineas) == 0:
            st.warning("El archivo diccionario_variables_ddc.txt está vacío.")
        else:
            for linea in lineas:
                if ":" in linea:
                    variable, descripcion = linea.split(":", 1)
                    st.markdown(
                        f"""
                        <div class="bloque">
                        <b>{variable.strip()}</b><br>
                        {descripcion.strip()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="bloque">
                        {linea}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    except FileNotFoundError:
        st.warning("No se encontró el archivo diccionario_variables_ddc.txt.")

# ============================================================
# BIBLIOGRAFÍA
# ============================================================

elif menu == "Bibliografía":

    st.header("Bibliografía")

    st.markdown(
        """
        <div class="bloque-azul">
        Referencias clínicas y académicas utilizadas para contextualizar la Displasia del
        Desarrollo de Cadera y el uso de herramientas de Machine Learning en el proyecto.
        </div>
        """,
        unsafe_allow_html=True
    )

    mostrar_bibliografia()

# ============================================================
# INTERPRETACIÓN CLÍNICA
# ============================================================

elif menu == "Interpretación clínica":

    st.header("Interpretación clínica educativa")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            <div class="bloque-azul">
            <b>SOPP+IA Sofía</b> es una herramienta educativa de inteligencia aumentada.
            Su función es apoyar el análisis de factores de riesgo de Displasia del Desarrollo
            de Cadera.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Lectura sugerida del resultado")

        st.markdown(
            """
            <div class="bloque">
            <b>Riesgo bajo:</b><br>
            Mantener vigilancia clínica según edad, antecedentes y signos físicos.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="bloque">
            <b>Riesgo intermedio:</b><br>
            Correlacionar con examen físico y estudio de imagen.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="bloque">
            <b>Riesgo alto:</b><br>
            Recomienda evaluación prioritaria por Ortopedia Pediátrica.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.warning(
            "Este predictor no reemplaza ecografía, radiografía, examen físico ni criterio clínico."
        )

    with col2:
        try:
            st.image(ICONO, caption="SOPP+IA Sofía", use_container_width=True)
        except Exception:
            st.info("SOPP+IA Sofía")

# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown(
    """
    <div class="footer">
    SOPP+IA Sofía · Inteligencia humana + inteligencia artificial para servir mejor a niños y familias
    </div>
    """,
    unsafe_allow_html=True
)
