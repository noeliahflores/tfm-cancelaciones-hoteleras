from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="TFM | Cancelaciones hoteleras",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos"
MODEL_DIR = BASE_DIR / "modelo"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
MONTH_TO_NUM = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================


def formato_euro(valor):
    """Da formato de euros con separador decimal español."""
    try:
        valor = float(valor)
        return f"{valor:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def formato_numero(valor, decimales=0):
    try:
        valor = float(valor)
        return f"{valor:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "-"


def formato_porcentaje(valor):
    """Acepta porcentajes en escala 0-1 o 0-100."""
    try:
        valor = float(valor)
        if abs(valor) <= 1:
            valor *= 100
        return f"{valor:.2f}%".replace(".", ",")
    except Exception:
        return "-"


def clasificar_riesgo(probabilidad):
    if probabilidad < 0.30:
        return "Riesgo bajo", "La reserva presenta una probabilidad reducida de cancelación. No sería necesaria una actuación especial."
    if probabilidad < 0.60:
        return "Riesgo medio", "Sería recomendable hacer seguimiento de la reserva, enviar recordatorios o revisar las condiciones de pago."
    return "Riesgo alto", "La reserva tiene una probabilidad elevada de cancelación. El hotel podría solicitar confirmación, revisar el depósito o considerarla dentro de una estrategia de overbooking controlado."


def leer_csv(nombre):
    ruta = DATA_DIR / nombre
    if ruta.exists():
        return pd.read_csv(ruta)
    return pd.DataFrame()


def valor_resumen(resumen_df, indicador, defecto=np.nan):
    try:
        fila = resumen_df.loc[resumen_df["indicador"] == indicador, "valor"]
        if len(fila) > 0:
            return float(fila.iloc[0])
    except Exception:
        pass
    return defecto


def opciones_columna(df, columna, top_n=None):
    if columna not in df.columns:
        return []
    serie = df[columna].dropna().astype(str)
    if top_n:
        return serie.value_counts().head(top_n).index.tolist()
    return sorted(serie.unique().tolist())


def media_entera(df, columna, defecto):
    if columna in df.columns and df[columna].notna().any():
        return int(round(df[columna].median()))
    return defecto


def media_float(df, columna, defecto):
    if columna in df.columns and df[columna].notna().any():
        return float(round(df[columna].median(), 2))
    return defecto


def grafico_barras(df, x, y, titulo, labels=None, text_auto=True):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No hay datos suficientes para mostrar este gráfico.")
        return
    fig = px.bar(df, x=x, y=y, text_auto=text_auto, title=titulo, labels=labels or {})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================


@st.cache_data
def cargar_datos():
    df = leer_csv("hotel_bookings_dashboard.csv")
    resumen = leer_csv("resumen_economico.csv")
    impacto_hotel = leer_csv("impacto_por_hotel.csv")
    overbooking = leer_csv("overbooking_por_hotel.csv")
    impacto_modelo = leer_csv("impacto_modelo.csv")
    metricas = leer_csv("metricas_modelos.csv")
    escenarios = leer_csv("escenarios_retorno.csv")
    return df, resumen, impacto_hotel, overbooking, impacto_modelo, metricas, escenarios


@st.cache_resource
def cargar_modelo():
    """Carga los objetos entrenados en el notebook."""
    resultado = {
        "modelo": None,
        "preprocesador": None,
        "variables": None,
        "umbral": 0.5,
        "metadata": {},
        "error": None,
    }
    try:
        resultado["modelo"] = joblib.load(MODEL_DIR / "modelo_cancelaciones.pkl")
        resultado["preprocesador"] = joblib.load(MODEL_DIR / "preprocesador_cancelaciones.pkl")
        resultado["variables"] = joblib.load(MODEL_DIR / "variables_modelo.pkl")
        resultado["umbral"] = joblib.load(MODEL_DIR / "umbral_decision.pkl")
        ruta_metadata = MODEL_DIR / "metadata_modelo.json"
        if ruta_metadata.exists():
            with open(ruta_metadata, "r", encoding="utf-8") as f:
                resultado["metadata"] = json.load(f)
    except Exception as e:
        resultado["error"] = str(e)
    return resultado


# ============================================================
# LECTURA INICIAL
# ============================================================


df, resumen, impacto_hotel, overbooking_df, impacto_modelo, metricas_modelos, escenarios_retorno = cargar_datos()
objetos_modelo = cargar_modelo()

# ============================================================
# SIDEBAR
# ============================================================

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <h2 style="margin-bottom: 0;"> Sistema predictivo</h2>
        <h3 style="margin-top: 0; color: #4A5568;">Cancelaciones hoteleras</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Trabajo Fin de Máster**

    Prototipo interactivo para analizar reservas hoteleras, predecir cancelaciones y estimar su impacto económico.
    """
)

st.sidebar.markdown("### Modelo utilizado")

if not metricas_modelos.empty:
    mejor_modelo = metricas_modelos.iloc[0]

    st.sidebar.markdown(
        f"""
        <div style="
            background-color: #FFFFFF;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #E5E7EB;
            margin-bottom: 15px;
        ">
            <p style="margin-bottom: 5px; color: #6B7280;">Modelo final</p>
            <h3 style="margin-top: 0;">{mejor_modelo['modelo']}</h3>
            <p><b>ROC-AUC:</b> {mejor_modelo['roc_auc']:.4f}</p>
            <p><b>F1-score:</b> {mejor_modelo['f1_score']:.4f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if objetos_modelo["error"]:
    st.sidebar.warning("El dashboard carga los datos correctamente, pero no se ha podido cargar el modelo de predicción.")

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    <small>
    TFM desarrollado por Noelia Hernández Flores.
    </small>
    """,
    unsafe_allow_html=True
)

# ============================================================
# TÍTULO PRINCIPAL
# ============================================================

st.title("Predicción de cancelaciones hoteleras")
st.markdown(
    """
    Esta demo muestra cómo el modelo desarrollado en el notebook puede utilizarse en un contexto práctico. La aplicación combina análisis descriptivo, predicción de cancelaciones e interpretación económica para apoyar la toma de decisiones hoteleras.
    """
)

if df.empty:
    st.error("No se ha encontrado el archivo `hotel_bookings_dashboard.csv` dentro de la carpeta `datos`.")
    st.stop()

# ============================================================
# PESTAÑAS DE LA APP
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "1. Análisis hotelero",
    "2. Predicción de cancelación",
    "3. Impacto económico y overbooking"
])

# ============================================================
# TAB 1: DASHBOARD HOTELERO
# ============================================================

with tab1:
    st.header("1. Dashboard de análisis hotelero")
    st.markdown(
        """
        En esta sección se resumen los principales patrones del conjunto de datos: volumen de reservas, porcentaje de cancelaciones, evolución mensual, tipo de hotel y canales de reserva con mayor riesgo.
        """
    )

    col_f1, col_f2, col_f3 = st.columns(3)
    hoteles = ["Todos"] + opciones_columna(df, "hotel")
    años = ["Todos"] + sorted(df["arrival_date_year"].dropna().unique().astype(int).tolist())

    with col_f1:
        hotel_sel = st.selectbox("Filtrar por tipo de hotel", hoteles)
    with col_f2:
        año_sel = st.selectbox("Filtrar por año", años)
    with col_f3:
        mostrar_datos = st.checkbox("Mostrar tabla de datos", value=False)

    df_filtrado = df.copy()
    if hotel_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["hotel"] == hotel_sel]
    if año_sel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["arrival_date_year"] == año_sel]

    total_reservas = len(df_filtrado)
    total_canceladas = int(df_filtrado["is_canceled"].sum())
    tasa_cancelacion = df_filtrado["is_canceled"].mean() if total_reservas else 0
    adr_medio = df_filtrado["adr"].mean() if total_reservas else 0
    ingreso_total = df_filtrado["ingreso_estimado_reserva"].sum() if "ingreso_estimado_reserva" in df_filtrado.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reservas totales", formato_numero(total_reservas, 0))
    c2.metric("Reservas canceladas", formato_numero(total_canceladas, 0))
    c3.metric("Tasa de cancelación", formato_porcentaje(tasa_cancelacion))
    c4.metric("ADR medio", formato_euro(adr_medio))

    st.metric("Ingreso estimado del filtro seleccionado", formato_euro(ingreso_total))

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        cancelaciones = (
            df_filtrado["is_canceled"]
            .map({0: "No cancelada", 1: "Cancelada"})
            .value_counts()
            .reset_index()
        )
        cancelaciones.columns = ["estado", "reservas"]
        fig = px.pie(cancelaciones, names="estado", values="reservas", title="Distribución de reservas canceladas y no canceladas")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        por_hotel = (
            df_filtrado.groupby("hotel", as_index=False)
            .agg(reservas=("is_canceled", "count"), tasa_cancelacion=("is_canceled", "mean"))
        )
        por_hotel["tasa_cancelacion_pct"] = por_hotel["tasa_cancelacion"] * 100
        grafico_barras(
            por_hotel,
            x="hotel",
            y="tasa_cancelacion_pct",
            titulo="Tasa de cancelación por tipo de hotel",
            labels={"hotel": "Tipo de hotel", "tasa_cancelacion_pct": "Tasa de cancelación (%)"},
        )

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        if "arrival_date_month" in df_filtrado.columns:
            mensual = (
                df_filtrado.groupby(["arrival_date_month", "arrival_month_num"], as_index=False)
                .agg(reservas=("is_canceled", "count"), cancelaciones=("is_canceled", "sum"))
                .sort_values("arrival_month_num")
            )
            mensual["tasa_cancelacion"] = mensual["cancelaciones"] / mensual["reservas"] * 100
            fig = px.line(
                mensual,
                x="arrival_date_month",
                y="tasa_cancelacion",
                markers=True,
                title="Evolución mensual de la tasa de cancelación",
                labels={"arrival_date_month": "Mes", "tasa_cancelacion": "Tasa de cancelación (%)"},
                category_orders={"arrival_date_month": MONTH_ORDER},
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_g4:
        canal = (
            df_filtrado.groupby("market_segment", as_index=False)
            .agg(reservas=("is_canceled", "count"), tasa_cancelacion=("is_canceled", "mean"))
            .sort_values("tasa_cancelacion", ascending=False)
        )
        canal["tasa_cancelacion_pct"] = canal["tasa_cancelacion"] * 100
        grafico_barras(
            canal,
            x="market_segment",
            y="tasa_cancelacion_pct",
            titulo="Tasa de cancelación por segmento de mercado",
            labels={"market_segment": "Segmento", "tasa_cancelacion_pct": "Tasa de cancelación (%)"},
        )

    col_g5, col_g6 = st.columns(2)

    with col_g5:
        deposito = (
            df_filtrado.groupby("deposit_type", as_index=False)
            .agg(reservas=("is_canceled", "count"), tasa_cancelacion=("is_canceled", "mean"))
            .sort_values("tasa_cancelacion", ascending=False)
        )
        deposito["tasa_cancelacion_pct"] = deposito["tasa_cancelacion"] * 100
        grafico_barras(
            deposito,
            x="deposit_type",
            y="tasa_cancelacion_pct",
            titulo="Tasa de cancelación por tipo de depósito",
            labels={"deposit_type": "Tipo de depósito", "tasa_cancelacion_pct": "Tasa de cancelación (%)"},
        )

    with col_g6:
        cliente = (
            df_filtrado.groupby("customer_type", as_index=False)
            .agg(reservas=("is_canceled", "count"), tasa_cancelacion=("is_canceled", "mean"))
            .sort_values("tasa_cancelacion", ascending=False)
        )
        cliente["tasa_cancelacion_pct"] = cliente["tasa_cancelacion"] * 100
        grafico_barras(
            cliente,
            x="customer_type",
            y="tasa_cancelacion_pct",
            titulo="Tasa de cancelación por tipo de cliente",
            labels={"customer_type": "Tipo de cliente", "tasa_cancelacion_pct": "Tasa de cancelación (%)"},
        )

    st.subheader("Variables más relevantes del modelo")
    st.markdown(
        """
        A continuación se muestran las variables con mayor importancia global para el modelo. Esta importancia no explica una reserva concreta de forma individual, pero ayuda a entender qué factores han tenido más peso en el aprendizaje del modelo.
        """
    )

    modelo = objetos_modelo.get("modelo")
    preprocesador = objetos_modelo.get("preprocesador")
    variables = objetos_modelo.get("variables")

    if modelo is not None and preprocesador is not None and variables is not None and hasattr(modelo, "feature_importances_"):
        try:
            nombres = preprocesador.get_feature_names_out(variables)
            importancias = pd.DataFrame({"variable": nombres, "importancia": modelo.feature_importances_})
            importancias["variable"] = (
                importancias["variable"]
                .str.replace("num__", "", regex=False)
                .str.replace("cat__", "", regex=False)
            )
            importancias = importancias.sort_values("importancia", ascending=False).head(15)
            fig = px.bar(
                importancias.sort_values("importancia"),
                x="importancia",
                y="variable",
                orientation="h",
                title="Top 15 variables más importantes del modelo",
                labels={"importancia": "Importancia", "variable": "Variable"},
            )
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("No ha sido posible calcular automáticamente la importancia de variables en esta ejecución.")
    else:
        st.info("La importancia de variables estará disponible cuando el modelo y el preprocesador se carguen correctamente.")

    if mostrar_datos:
        st.subheader("Vista previa del dataset")
        st.dataframe(df_filtrado.head(100), use_container_width=True)

# ============================================================
# TAB 2: PREDICCIÓN
# ============================================================

with tab2:
    st.header("2. Predicción de cancelación en tiempo real")
    st.markdown(
        """
        En esta sección se introducen las características de una nueva reserva y el modelo devuelve una probabilidad estimada de cancelación. A partir de esa probabilidad, la reserva se clasifica como riesgo bajo, medio o alto.
        """
    )

    if objetos_modelo["error"]:
        st.error("No se ha podido cargar el modelo o el preprocesador.")
        st.code(objetos_modelo["error"])
        st.info("Revisa que los archivos `.pkl` estén dentro de la carpeta `modelo` y que las versiones de `scikit-learn` y `xgboost` coincidan con las indicadas en `requirements.txt`.")
    else:
        variables_modelo = objetos_modelo["variables"]
        umbral = float(objetos_modelo["umbral"])

        with st.form("formulario_prediccion"):
            st.subheader("Datos principales de la reserva")
            col1, col2, col3 = st.columns(3)

            with col1:
                hotel = st.selectbox("Tipo de hotel", opciones_columna(df, "hotel"))
                lead_time = st.number_input("Antelación de la reserva (días)", min_value=0, max_value=800, value=media_entera(df, "lead_time", 70))
                mes = st.selectbox("Mes de llegada", MONTH_ORDER, index=MONTH_ORDER.index("August"))
                adr = st.number_input("Precio medio por noche (ADR)", min_value=0.0, max_value=1000.0, value=media_float(df, "adr", 100.0), step=5.0)

            with col2:
                market_segment = st.selectbox("Segmento de mercado", opciones_columna(df, "market_segment"), index=opciones_columna(df, "market_segment").index("Online TA") if "Online TA" in opciones_columna(df, "market_segment") else 0)
                distribution_channel = st.selectbox("Canal de distribución", opciones_columna(df, "distribution_channel"))
                customer_type = st.selectbox("Tipo de cliente", opciones_columna(df, "customer_type"))
                deposit_type = st.selectbox("Tipo de depósito", opciones_columna(df, "deposit_type"))

            with col3:
                stays_in_weekend_nights = st.number_input("Noches de fin de semana", min_value=0, max_value=30, value=1)
                stays_in_week_nights = st.number_input("Noches entre semana", min_value=0, max_value=60, value=2)
                adults = st.number_input("Adultos", min_value=0, max_value=10, value=2)
                children = st.number_input("Niños", min_value=0, max_value=10, value=0)

            st.subheader("Datos adicionales")
            with st.expander("Mostrar variables adicionales del modelo"):
                col4, col5, col6 = st.columns(3)
                with col4:
                    arrival_date_year = st.selectbox("Año de llegada", sorted(df["arrival_date_year"].dropna().unique().astype(int).tolist()), index=len(sorted(df["arrival_date_year"].dropna().unique().astype(int).tolist())) - 1)
                    arrival_date_week_number = st.number_input("Semana del año", min_value=1, max_value=53, value=media_entera(df, "arrival_date_week_number", 27))
                    arrival_date_day_of_month = st.number_input("Día del mes", min_value=1, max_value=31, value=15)
                    meal = st.selectbox("Régimen", opciones_columna(df, "meal"))
                with col5:
                    paises = opciones_columna(df, "country")
                    country_default = paises.index("PRT") if "PRT" in paises else 0
                    country = st.selectbox("País", paises, index=country_default)
                    reserved_room_type = st.selectbox("Tipo de habitación reservada", opciones_columna(df, "reserved_room_type"))
                    babies = st.number_input("Bebés", min_value=0, max_value=10, value=0)
                    is_repeated_guest = st.selectbox("Cliente repetido", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
                with col6:
                    previous_cancellations = st.number_input("Cancelaciones previas", min_value=0, max_value=30, value=0)
                    previous_bookings_not_canceled = st.number_input("Reservas previas no canceladas", min_value=0, max_value=100, value=0)
                    required_car_parking_spaces = st.number_input("Plazas de parking solicitadas", min_value=0, max_value=8, value=0)
                    total_of_special_requests = st.number_input("Peticiones especiales", min_value=0, max_value=5, value=0)

            boton = st.form_submit_button("Predecir cancelación")

        if boton:
            total_nights = int(stays_in_weekend_nights + stays_in_week_nights)
            total_guests = int(adults + children + babies)
            ingreso_reserva = float(adr * total_nights)

            reserva = pd.DataFrame([{
                "hotel": hotel,
                "lead_time": int(lead_time),
                "arrival_date_year": int(arrival_date_year),
                "arrival_month_num": int(MONTH_TO_NUM[mes]),
                "arrival_date_week_number": int(arrival_date_week_number),
                "arrival_date_day_of_month": int(arrival_date_day_of_month),
                "total_nights": int(total_nights),
                "stays_in_weekend_nights": int(stays_in_weekend_nights),
                "stays_in_week_nights": int(stays_in_week_nights),
                "total_guests": int(total_guests),
                "adults": int(adults),
                "children": int(children),
                "babies": int(babies),
                "meal": meal,
                "country": country,
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": int(is_repeated_guest),
                "previous_cancellations": int(previous_cancellations),
                "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
                "reserved_room_type": reserved_room_type,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "adr": float(adr),
                "required_car_parking_spaces": int(required_car_parking_spaces),
                "total_of_special_requests": int(total_of_special_requests),
            }])

            try:
                reserva = reserva[variables_modelo]
                reserva_preprocesada = objetos_modelo["preprocesador"].transform(reserva)
                prob_cancelacion = float(objetos_modelo["modelo"].predict_proba(reserva_preprocesada)[0, 1])
                prediccion = int(prob_cancelacion >= umbral)
                riesgo, recomendacion = clasificar_riesgo(prob_cancelacion)
                ingreso_en_riesgo = ingreso_reserva * prob_cancelacion

                st.subheader("Resultado de la predicción")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Probabilidad de cancelación", formato_porcentaje(prob_cancelacion))
                r2.metric("Nivel de riesgo", riesgo)
                r3.metric("Ingreso estimado", formato_euro(ingreso_reserva))
                r4.metric("Ingreso en riesgo", formato_euro(ingreso_en_riesgo))

                if prediccion == 1:
                    st.warning("Según el umbral definido en el notebook, el modelo clasifica esta reserva como **posible cancelación**.")
                else:
                    st.success("Según el umbral definido en el notebook, el modelo clasifica esta reserva como **probable no cancelación**.")

                st.markdown(f"**Recomendación para el hotel:** {recomendacion}")

                st.subheader("Factores interpretativos de la reserva")
                factores = []
                if lead_time >= df["lead_time"].quantile(0.75):
                    factores.append("Antelación elevada de la reserva")
                if deposit_type != "No Deposit":
                    factores.append("Tipo de depósito diferente a 'No Deposit'")
                if total_of_special_requests == 0:
                    factores.append("No hay peticiones especiales")
                if previous_cancellations > 0:
                    factores.append("Existen cancelaciones previas")
                if market_segment in ["Groups", "Online TA", "Offline TA/TO"]:
                    factores.append("Segmento de mercado asociado históricamente a mayor riesgo")
                if adr >= df["adr"].quantile(0.75):
                    factores.append("Precio medio por noche elevado")

                if factores:
                    for f in factores:
                        st.write(f"- {f}")
                else:
                    st.write("No se observan factores de riesgo especialmente destacados en los campos introducidos.")

                with st.expander("Ver datos enviados al modelo"):
                    st.dataframe(reserva, use_container_width=True)

            except Exception as e:
                st.error("No se ha podido realizar la predicción con los datos introducidos.")
                st.code(str(e))

# ============================================================
# TAB 3: IMPACTO ECONÓMICO
# ============================================================

with tab3:
    st.header("3. Estimación económica y overbooking controlado")
    st.markdown(
        """
        Esta sección transforma la predicción de cancelaciones en una lectura económica. El objetivo es estimar cuánto ingreso potencial está asociado a las cancelaciones y cómo podría utilizarse el modelo para apoyar una estrategia de overbooking controlado.
        """
    )

    total_reservas_res = valor_resumen(resumen, "Total de reservas", len(df))
    tasa_cancelacion_res = valor_resumen(resumen, "Tasa de cancelación", df["is_canceled"].mean())
    ingreso_total_res = valor_resumen(resumen, "Ingreso total estimado", df["ingreso_estimado_reserva"].sum())
    ingreso_cancelado_res = valor_resumen(resumen, "Ingreso potencial asociado a cancelaciones", df.loc[df["is_canceled"] == 1, "ingreso_estimado_reserva"].sum())
    porcentaje_ingreso_cancelado = valor_resumen(resumen, "% del ingreso estimado asociado a cancelaciones", np.nan)

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Reservas analizadas", formato_numero(total_reservas_res, 0))
    e2.metric("Tasa de cancelación", formato_porcentaje(tasa_cancelacion_res))
    e3.metric("Ingreso total estimado", formato_euro(ingreso_total_res))
    e4.metric("Ingreso asociado a cancelaciones", formato_euro(ingreso_cancelado_res))

    st.info(
        f"En el conjunto de datos, las cancelaciones representan aproximadamente **{formato_porcentaje(porcentaje_ingreso_cancelado)}** del ingreso estimado total."
    )

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        if not impacto_hotel.empty:
            fig = px.bar(
                impacto_hotel,
                x="hotel",
                y="ingreso_cancelaciones",
                text_auto=True,
                title="Ingreso potencial asociado a cancelaciones por tipo de hotel",
                labels={"hotel": "Tipo de hotel", "ingreso_cancelaciones": "Ingreso asociado a cancelaciones"},
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(impacto_hotel, use_container_width=True)

    with col_e2:
        if not overbooking_df.empty:
            fig = px.bar(
                overbooking_df,
                x="hotel",
                y="reservas_extra_estimadas",
                text_auto=True,
                title="Reservas extra estimadas por cada 100 habitaciones",
                labels={"hotel": "Tipo de hotel", "reservas_extra_estimadas": "Reservas extra"},
            )
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(overbooking_df, use_container_width=True)

    st.subheader("Impacto económico del modelo en el conjunto de test")
    if not impacto_modelo.empty:
        fig = px.bar(
            impacto_modelo,
            x="tipo_resultado",
            y="ingreso_estimado",
            text_auto=True,
            title="Ingreso estimado según tipo de resultado del modelo",
            labels={"tipo_resultado": "Tipo de resultado", "ingreso_estimado": "Ingreso estimado"},
        )
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=60, b=120), xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(impacto_modelo, use_container_width=True)

    st.subheader("Escenarios de retorno potencial")
    if not escenarios_retorno.empty:
        fig = px.bar(
            escenarios_retorno,
            x="escenario_recuperacion",
            y="retorno_potencial_estimado",
            text_auto=True,
            title="Retorno potencial según porcentaje de recuperación de ingresos en riesgo",
            labels={"escenario_recuperacion": "Escenario", "retorno_potencial_estimado": "Retorno potencial estimado"},
        )
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(escenarios_retorno, use_container_width=True)

    st.subheader("Simulador de punto de equilibrio de overbooking")
    st.markdown(
        """
        El punto de equilibrio indica cuántas reservas habría que aceptar para intentar llenar una capacidad determinada, considerando una tasa esperada de cancelación.
        """
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        capacidad = st.number_input("Capacidad simulada de habitaciones", min_value=1, max_value=1000, value=100)
    with col_s2:
        tasa_manual = st.slider("Tasa esperada de cancelación", min_value=0.0, max_value=0.9, value=float(round(tasa_cancelacion_res, 2)), step=0.01)
    with col_s3:
        ingreso_medio = st.number_input("Ingreso medio por reserva no cancelada", min_value=0.0, max_value=5000.0, value=float(round(df.loc[df["is_canceled"] == 0, "ingreso_estimado_reserva"].mean(), 2)), step=10.0)

    if tasa_manual < 1:
        reservas_necesarias = math.ceil(capacidad / (1 - tasa_manual))
        reservas_extra = reservas_necesarias - capacidad
        habitaciones_vacias_sin_overbooking = capacidad * tasa_manual
        ingreso_adicional_potencial = habitaciones_vacias_sin_overbooking * ingreso_medio

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Reservas necesarias", formato_numero(reservas_necesarias, 0))
        s2.metric("Reservas extra", formato_numero(reservas_extra, 0))
        s3.metric("Habitaciones vacías sin overbooking", formato_numero(habitaciones_vacias_sin_overbooking, 2))
        s4.metric("Ingreso adicional potencial", formato_euro(ingreso_adicional_potencial))

        st.markdown(
            f"""
            Con una capacidad de **{capacidad} habitaciones** y una tasa esperada de cancelación del **{formato_porcentaje(tasa_manual)}**, el hotel tendría que aceptar aproximadamente **{reservas_necesarias} reservas** para alcanzar el punto de equilibrio. Esto equivale a **{reservas_extra} reservas extra**.
            """
        )

    st.caption("La estimación económica es aproximada y debe interpretarse como apoyo a la toma de decisiones, no como una regla automática de gestión hotelera.")
