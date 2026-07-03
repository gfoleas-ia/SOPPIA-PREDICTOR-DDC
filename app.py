
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DATASET = "dataset_sintetico_ddc_graf_auditado_v2.csv"
DICCIONARIO = "diccionario_variables_ddc_graf_auditado_v2.txt"
ICONO = "assets/soppia_sofia_icon.png"
OBJETIVO = "ddc_diagnostico"

try:
    icono_sofia = Image.open(ICONO)
except Exception:
    icono_sofia = "🦴"

st.set_page_config(
    page_title="SOPP+IA Sofía | Predictor educativo DDC",
    page_icon=icono_sofia,
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0057B8, #003B7A);
}
[data-testid="stSidebar"] * {
    color: white;
}
.titulo {
    color: #003B7A;
    font-size: 34px;
    font-weight: 800;
    line-height: 1.2;
}
.subtitulo {
    color: #334155;
    font-size: 18px;
    line-height: 1.5;
}
.alerta {
    background-color: #E8F2FF;
    color: #003B7A;
    padding: 16px;
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
}
.numero {
    color: #003B7A;
    font-size: 30px;
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
    padding: 18px;
    background-color: #E8F2FF;
    border-radius: 10px;
    text-align: center;
    color: #003B7A;
    font-weight: 600;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


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
                "NO": "No",
                "no": "No",
                "No ": "No"
            })
        )

    return df


@st.cache_resource
def entrenar_modelo(df):
    datos = df.copy()

    columnas_no_modelo = [
        "id_paciente",
        "riesgo_teorico_ddc",
        "resultado_final_simulado"
    ]

    datos = datos.drop(columns=columnas_no_modelo, errors="ignore")

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

    matriz = confusion_matrix(y_test, y_pred, labels=["No", "Si"])

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
        return "Riesgo alto", "Se recomienda evaluación prioritaria por médico especialista en Ortopedia Pediátrica."
    elif probabilidad >= 0.40:
        return "Riesgo intermedio", "Se recomienda valoración clínica, examen físico y estudios de imagen según criterio del ortopedista pediatra."
    else:
        return "Riesgo bajo", "Mantener vigilancia. La evaluación médica sigue siendo necesaria si existen signos clínicos o antecedentes de riesgo."


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
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830."
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


try:
    df = cargar_datos()
except Exception:
    st.error("No se encontró el archivo dataset_sintetico_ddc_graf_auditado_v2.csv.")
    st.write("Verifique que el archivo esté cargado junto a app.py.")
    st.stop()

if OBJETIVO not in df.columns:
    st.error("El dataset no contiene la variable objetivo ddc_diagnostico.")
    st.stop()

modelo, X, metricas, matriz = entrenar_modelo(df)


try:
    st.sidebar.image(ICONO, width=150)
except Exception:
    st.sidebar.write("🦴")

st.sidebar.title("SOPP+IA Sofía")
st.sidebar.write("Predictor educativo de DDC")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Para padres y cuidadores**  

Aplicación para concientizar sobre el diagnóstico temprano y el tratamiento oportuno de la Displasia del Desarrollo de Cadera.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Desarrollado por:**  
Dr. Geovanny F. Oleas-Santillán  

Ortopedista Pediatra  
Quito, Ecuador
""")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Exploración de datos",
        "Ecografía Graf",
        "Modelo predictivo",
        "Predicción individual",
        "Diccionario",
        "Bibliografía",
        "Interpretación para padres"
    ]
)


if menu == "Inicio":

    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown(
            """
            <div class="titulo">
            SOPP+IA Sofía:<br>
            Simulador del Diagnóstico de la Displasia del Desarrollo de Cadera (DDC)<br>
            Con métricas de un dataset sintético. No equivalen a validación clínica externa, no se recomienda aplicarlo en pacientes reales.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <p class="subtitulo">
            Aplicación académica para entrenamiento de padres, madres y cuidadores de niños.
            Su objetivo es concientizar sobre la importancia del diagnóstico temprano y el
            tratamiento oportuno de la Displasia del Desarrollo de Cadera.
            </p>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="alerta">
            Esta aplicación es educativa. No reemplaza el examen físico, la ecografía, la radiografía
            ni la evaluación del médico especialista ortopedista pediatra.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="bloque-azul">
            <b>Desarrollado por:</b><br>
            Dr. Geovanny F. Oleas-Santillán<br>
            Ortopedista Pediatra · Quito, Ecuador<br>
            www.drgeovannyoleas.com
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        try:
            st.image(ICONO, caption="SOPP+IA Sofía", use_container_width=True)
        except Exception:
            st.info("SOPP+IA Sofía")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Pacientes sintéticos", df.shape[0])

    with c2:
        card("Variables", df.shape[1])

    with c3:
        card("Casos DDC", int((df[OBJETIVO] == "Si").sum()))

    with c4:
        card("Modelo", "Random Forest")

    st.subheader("Distribución del diagnóstico")

    fig, ax = plt.subplots(figsize=(6, 4))
    df[OBJETIVO].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Diagnóstico DDC")
    ax.set_ylabel("Número de pacientes")
    ax.set_title("Distribución de la variable objetivo")
    st.pyplot(fig)


elif menu == "Exploración de datos":

    st.header("Exploración del dataset auditado")

    resumen = pd.DataFrame({
        "Indicador": [
            "Pacientes",
            "Variables",
            "Datos faltantes",
            "DDC Sí",
            "DDC No"
        ],
        "Valor": [
            df.shape[0],
            df.shape[1],
            int(df.isnull().sum().sum()),
            int((df[OBJETIVO] == "Si").sum()),
            int((df[OBJETIVO] == "No").sum())
        ]
    })

    st.table(resumen)

    st.subheader("Primeras filas")
    st.dataframe(df.head(15), use_container_width=True)

    st.subheader("Resultados clínicos simulados")

    if "resultado_final_simulado" in df.columns:
        st.dataframe(
            df[
                [
                    "resultado_final_simulado",
                    "ddc_diagnostico"
                ]
            ].head(20),
            use_container_width=True
        )

    st.subheader("Visualización por variable")

    variable = st.selectbox(
        "Seleccione una variable",
        [c for c in df.columns if c != "id_paciente"]
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    if pd.api.types.is_numeric_dtype(df[variable]):
        ax.hist(df[variable].dropna())
        ax.set_title(f"Histograma de {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Frecuencia")
    else:
        df[variable].astype(str).value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Distribución de {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Frecuencia")

    st.pyplot(fig)


elif menu == "Ecografía Graf":

    st.header("Variables ecográficas según método de Graf")

    st.markdown(
        """
        <div class="bloque-azul">
        El método de Graf clasifica cada cadera por separado. Por eso el dataset contiene
        grupo_graf_derecho y grupo_graf_izquierdo, sin usar una clasificación global.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Clasificación ecográfica basada en los principios descritos por Reinhard Graf para el diagnóstico temprano de la Displasia del Desarrollo de Cadera."
    )

    variables_graf = [
        "angulo_alfa_derecho",
        "angulo_alfa_izquierdo",
        "angulo_beta_derecho",
        "angulo_beta_izquierdo",
        "cobertura_cabeza_femoral_derecha_pct",
        "cobertura_cabeza_femoral_izquierda_pct",
        "grupo_graf_derecho",
        "grupo_graf_izquierdo"
    ]

    st.dataframe(df[variables_graf].head(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Grupo Graf derecho")
        fig, ax = plt.subplots()
        df["grupo_graf_derecho"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Grupo Graf")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)

    with col2:
        st.subheader("Grupo Graf izquierdo")
        fig, ax = plt.subplots()
        df["grupo_graf_izquierdo"].value_counts().plot(kind="bar", ax=ax)
        ax.set_xlabel("Grupo Graf")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)


elif menu == "Modelo predictivo":

    st.header("Modelo predictivo de Machine Learning")

    st.markdown(
        """
        <div class="bloque-azul">
        Se utiliza Random Forest Classifier. Las variables categóricas se transforman con
        One-Hot Encoding y las variables numéricas pasan directamente al modelo.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        metricas_df = pd.DataFrame({
            "Métrica": list(metricas.keys()),
            "Valor": [round(v, 3) for v in metricas.values()]
        })

        st.subheader("Métricas")
        st.table(metricas_df)

    with col2:
        matriz_df = pd.DataFrame(
            matriz,
            index=["Real No", "Real Si"],
            columns=["Predicho No", "Predicho Si"]
        )

        st.subheader("Matriz de confusión")
        st.table(matriz_df)

    st.warning("Estas métricas proceden de un dataset sintético. No equivalen a validación clínica externa.")


elif menu == "Predicción individual":

    st.header("Predicción individual educativa")

    st.markdown(
        """
        <div class="alerta">
        Complete los datos simulados. El resultado sirve para educación y entrenamiento.
        Siempre se recomienda evaluación por un médico especialista ortopedista pediatra.
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
                        columna,
                        value=float(X[columna].mean())
                    )
                else:
                    opciones = sorted(X[columna].astype(str).dropna().unique().tolist())
                    entrada[columna] = st.selectbox(columna, opciones)

        boton = st.form_submit_button("Analizar riesgo educativo de DDC")

    if boton:

        nuevo_paciente = pd.DataFrame([entrada])

        prediccion = modelo.predict(nuevo_paciente)[0]
        probabilidad = obtener_probabilidad_ddc(modelo, nuevo_paciente)
        categoria, recomendacion = clasificar_riesgo(probabilidad)

        st.subheader("Resultado educativo")

        r1, r2, r3 = st.columns(3)

        with r1:
            card("Clasificación", prediccion)

        with r2:
            card("Probabilidad DDC", f"{probabilidad:.1%}")

        with r3:
            card("Categoría", categoria)

        st.info(recomendacion)

        st.warning(
            "Este resultado no confirma ni descarta DDC. La decisión clínica corresponde al ortopedista pediatra."
        )

        with st.expander("Ver datos ingresados"):
            st.dataframe(nuevo_paciente, use_container_width=True)


elif menu == "Diccionario":

    st.header("Diccionario de variables")

    try:
        with open(DICCIONARIO, "r", encoding="utf-8") as f:
            texto_diccionario = f.read()

        st.text(texto_diccionario)

    except Exception:
        st.error("No se encontró el archivo diccionario_variables_ddc_graf_auditado_v2.txt.")


elif menu == "Bibliografía":

    st.header("Bibliografía clínica principal")
    mostrar_bibliografia()


elif menu == "Interpretación para padres":

    st.header("Información educativa para padres y cuidadores")

    st.markdown(
        """
        <div class="bloque-azul">
        La Displasia del Desarrollo de Cadera puede mejorar cuando se detecta temprano.
        El diagnóstico oportuno permite tratamientos menos invasivos y mejores resultados.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success(
        "El diagnóstico temprano durante los primeros meses de vida permite tratamientos menos invasivos y mejores resultados funcionales."
    )

    st.markdown("""
    ### Mensajes clave

    - La DDC puede presentarse aunque el niño parezca sano.
    - La ecografía de caderas es importante en los primeros meses de vida.
    - El método de Graf evalúa cada cadera por separado.
    - El diagnóstico tardío puede relacionarse con cojera, dolor, limitación funcional y artrosis temprana.
    - La evaluación debe ser realizada por un médico especialista en Ortopedia Pediátrica.
    """)

    st.warning(
        "Esta aplicación no reemplaza la consulta médica. Ante dudas, signos clínicos o factores de riesgo, acudir a Ortopedia Pediátrica."
    )


st.markdown(
    """
    <div class="footer">
        <b>SOPP+IA Sofía</b><br>
        Aplicación educativa para padres y cuidadores sobre diagnóstico temprano y tratamiento oportuno de la DDC.<br><br>
        Desarrollado por <b>Dr. Geovanny F. Oleas-Santillán</b><br>
        Ortopedista Pediatra · Quito, Ecuador<br>
        www.drgeovannyoleas.com
    </div>
    """,
    unsafe_allow_html=True
)
