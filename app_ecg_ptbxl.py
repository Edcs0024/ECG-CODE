import ast
import os
import io
import tempfile
import requests
import numpy as np
import pandas as pd
import wfdb
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, find_peaks


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

PTBXL_BASE = "ptb-xl/1.0.3"

st.set_page_config(
    page_title="Análisis ECG PTB-XL",
    layout="wide"
)


# ==========================================================
# CARGA DE METADATOS
# ==========================================================

@st.cache_data(ttl=3600, show_spinner="Cargando metadatos locales de PTB-XL...")
def cargar_metadatos():
    """Carga el CSV local; no descarga metadatos desde PhysioNet."""
    ruta_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptbxl_dataset_completo.csv")

    if not os.path.isfile(ruta_csv):
        st.error(
            "No se encontró ptbxl_dataset_completo.csv. "
            "Coloca ese archivo en la misma carpeta que app_ecg_ptbxl.py."
        )
        st.stop()

    try:
        db = pd.read_csv(ruta_csv, low_memory=False)
    except Exception as e:
        st.error(f"No se pudo leer el CSV local: {e}")
        st.stop()

    requeridas = {"ecg_id", "patient_id", "age", "sex", "scp_codes", "filename_lr"}
    faltantes = requeridas - set(db.columns)
    if faltantes:
        st.error("Al CSV le faltan columnas necesarias: " + ", ".join(sorted(faltantes)))
        st.stop()

    def parsear_codigos(valor):
        if isinstance(valor, dict):
            return valor
        if pd.isna(valor) or not str(valor).strip():
            return {}
        try:
            resultado = ast.literal_eval(str(valor))
            return resultado if isinstance(resultado, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    db["scp_codes"] = db["scp_codes"].apply(parsear_codigos)
    # El CSV contiene los códigos SCP pero no incluye scp_statements.csv,
    # que es el archivo que mapea cada código a una clase diagnóstica.
    # Para mantener los filtros Sano/Con afección usamos una regla explícita:
    # NORM sin otros códigos distintos de SR se considera Sano; los demás
    # registros con códigos se consideran Con afección. No es diagnóstico clínico.
    db["diagnostic_classes"] = db["scp_codes"].apply(lambda d: list(d.keys()))

    def clasificar_condicion(codigos):
        codigos = set(codigos.keys())
        if "NORM" in codigos and codigos.issubset({"NORM", "SR"}):
            return "Sano"
        if codigos:
            return "Con afección"
        return "No clasificado"

    db["condicion"] = db["scp_codes"].apply(clasificar_condicion)
    db["sexo_texto"] = pd.to_numeric(db["sex"], errors="coerce").map({0: "Hombre", 1: "Mujer"})
    db["age"] = pd.to_numeric(db["age"], errors="coerce")
    db["patient_id"] = pd.to_numeric(db["patient_id"], errors="coerce")
    db["ecg_id"] = pd.to_numeric(db["ecg_id"], errors="coerce")
    db = db.dropna(subset=["ecg_id", "patient_id", "age", "sexo_texto"])
    db = db.set_index("ecg_id", drop=False)
    db = db[(db["age"] >= 18) & (db["age"] <= 89)]
    db = db[db["condicion"].isin(["Sano", "Con afección"])]
    return db


# ==========================================================
# SELECCIÓN AUTOMÁTICA DE PACIENTES
# ==========================================================

def seleccionar_pacientes(db):  # No cachear el DataFrame: scp_codes contiene diccionarios no hashables.
    grupos = []

    rangos_edad = {
        "18-40": (18, 40),
        "60-89": (60, 89)
    }

    for sexo in ["Hombre", "Mujer"]:
        for rango, limites in rangos_edad.items():
            edad_min, edad_max = limites

            for condicion in ["Sano", "Con afección"]:
                subset = db[
                    (db["sexo_texto"] == sexo) &
                    (db["age"] >= edad_min) &
                    (db["age"] <= edad_max) &
                    (db["condicion"] == condicion)
                ]

                # Evitar repetir el mismo paciente
                subset = subset.drop_duplicates("patient_id").head(5)
                subset = subset.copy()
                subset["grupo"] = f"{sexo} {rango} {condicion}"

                grupos.append(subset)

    seleccion = pd.concat(grupos)

    return seleccion


# ==========================================================
# LECTURA DE ECG DESDE PHYSIONET
# ==========================================================

@st.cache_data(show_spinner="Descargando el ECG seleccionado...")
def leer_ecg_ptbxl(filename_lr):
    """
    Descarga y lee un registro de PTB-XL.
    Descarga el encabezado (.hea) y los datos (.dat) con reintentos,
    y guarda ambos usando el nombre de registro esperado por WFDB.
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    ruta_relativa = str(filename_lr).replace("\\\\", "/").lstrip("/")
    base_url = f"https://physionet.org/files/{PTBXL_BASE}/{ruta_relativa}"
    nombre_registro = os.path.basename(ruta_relativa)

    reintentos = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    sesion = requests.Session()
    sesion.mount("https://", HTTPAdapter(max_retries=reintentos))

    try:
        with tempfile.TemporaryDirectory(prefix="ecg_ptbxl_") as temp_dir:
            local_base = os.path.join(temp_dir, nombre_registro)

            for extension in (".hea", ".dat"):
                url = base_url + extension
                respuesta = sesion.get(url, timeout=(15, 90), stream=True)
                respuesta.raise_for_status()

                destino = local_base + extension
                bytes_escritos = 0
                with open(destino, "wb") as archivo:
                    for bloque in respuesta.iter_content(chunk_size=1024 * 256):
                        if bloque:
                            archivo.write(bloque)
                            bytes_escritos += len(bloque)
                respuesta.close()

                if bytes_escritos == 0:
                    raise IOError(f"PhysioNet devolvió un archivo vacío: {url}")

                # Si hay Content-Length, comprobar que la descarga llegó completa.
                longitud = respuesta.headers.get("Content-Length")
                if longitud and bytes_escritos != int(longitud):
                    raise IOError(
                        f"Descarga incompleta de {nombre_registro}{extension}: "
                        f"{bytes_escritos} de {longitud} bytes."
                    )

            if not os.path.isfile(local_base + ".hea") or not os.path.isfile(local_base + ".dat"):
                raise FileNotFoundError(
                    f"No se encontraron el encabezado y los datos de {nombre_registro}."
                )

            record = wfdb.rdrecord(local_base)

        return record.p_signal, record.fs, record.sig_name

    except requests.RequestException as exc:
        st.error(f"No se pudo descargar el ECG {filename_lr} desde PhysioNet: {exc}")
        st.info(
            "La señal requiere Internet. Vuelve a intentar o selecciona otro registro. "
            "Los metadatos sí se leen desde el CSV local."
        )
        st.stop()
    except Exception as exc:
        st.error(f"No se pudo leer el registro ECG {filename_lr}: {exc}")
        st.info(
            "Comprueba que la descarga del registro esté completa. "
            "Si el problema continúa, prueba con otra señal."
        )
        st.stop()


# ==========================================================
# FILTRADO PASA BANDA
# ==========================================================

def filtro_pasabanda(ecg, fs, lowcut=0.5, highcut=40, orden=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(orden, [low, high], btype="band")
    ecg_filtrado = filtfilt(b, a, ecg)

    return ecg_filtrado


# ==========================================================
# DETECCIÓN DE PICOS R
# ==========================================================

def detectar_picos_r(ecg_filtrado, fs):
    distancia_minima = int(0.4 * fs)
    altura_minima = np.mean(ecg_filtrado) + 0.6 * np.std(ecg_filtrado)

    picos, _ = find_peaks(
        ecg_filtrado,
        distance=distancia_minima,
        height=altura_minima
    )

    return picos


# ==========================================================
# SEGMENTACIÓN DE LATIDOS
# ==========================================================

def segmentar_latidos(ecg_filtrado, picos_r, fs, antes=0.25, despues=0.45):
    muestras_antes = int(antes * fs)
    muestras_despues = int(despues * fs)

    latidos = []

    for r in picos_r:
        inicio = r - muestras_antes
        fin = r + muestras_despues

        if inicio >= 0 and fin < len(ecg_filtrado):
            latidos.append(ecg_filtrado[inicio:fin])

    if len(latidos) == 0:
        return None, None, None

    latidos = np.array(latidos)
    latido_promedio = np.mean(latidos, axis=0)
    tiempo_latido = np.linspace(-antes, despues, len(latido_promedio))

    return latidos, latido_promedio, tiempo_latido


# ==========================================================
# VARIABILIDAD CARDÍACA
# ==========================================================

def calcular_hrv(picos_r, fs):
    if len(picos_r) < 3:
        return np.nan, np.nan, np.array([])

    rr_seg = np.diff(picos_r) / fs
    rr_ms = rr_seg * 1000

    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

    return sdnn, rmssd, rr_ms


# ==========================================================
# EXTRACCIÓN DE MÉTRICAS
# ==========================================================

def extraer_metricas(latido_promedio, tiempo_latido, fs, rr_ms, sexo):
    metricas = {}

    indice_r = np.argmax(latido_promedio)
    amplitud_r = latido_promedio[indice_r]

    # Estimar QRS
    umbral = 0.15 * np.max(np.abs(latido_promedio))

    inicio_qrs = indice_r
    while inicio_qrs > 0 and abs(latido_promedio[inicio_qrs]) > umbral:
        inicio_qrs -= 1

    fin_qrs = indice_r
    while fin_qrs < len(latido_promedio) - 1 and abs(latido_promedio[fin_qrs]) > umbral:
        fin_qrs += 1

    duracion_qrs = (fin_qrs - inicio_qrs) / fs

    # QT aproximado
    fin_qt = min(len(latido_promedio) - 1, indice_r + int(0.35 * fs))
    intervalo_qt = (fin_qt - inicio_qrs) / fs

    if len(rr_ms) > 0:
        rr_promedio = np.mean(rr_ms) / 1000
        qtc = intervalo_qt / np.sqrt(rr_promedio)
    else:
        qtc = np.nan

    # Onda Q aproximada
    inicio_busqueda_q = max(0, indice_r - int(0.08 * fs))
    segmento_q = latido_promedio[inicio_busqueda_q:indice_r]

    if len(segmento_q) > 0:
        q_min = np.min(segmento_q)
        indice_q = np.argmin(segmento_q) + inicio_busqueda_q
        profundidad_q = abs(q_min)
        duracion_q = (indice_r - indice_q) / fs
    else:
        profundidad_q = 0
        duracion_q = 0

    onda_q_patologica = (
        duracion_q > 0.04 and
        profundidad_q > 0.25 * abs(amplitud_r)
    )

    # Segmento ST
    punto_j = fin_qrs
    indice_st = min(len(latido_promedio) - 1, punto_j + int(0.06 * fs))

    zona_base = latido_promedio[max(0, inicio_qrs - int(0.08 * fs)):inicio_qrs]

    if len(zona_base) > 0:
        linea_base = np.mean(zona_base)
    else:
        linea_base = 0

    desviacion_st = latido_promedio[indice_st] - linea_base

    metricas["Duración QRS (ms)"] = duracion_qrs * 1000
    metricas["Intervalo QT (ms)"] = intervalo_qt * 1000
    metricas["QTc (ms)"] = qtc * 1000
    metricas["Amplitud R (mV)"] = amplitud_r
    metricas["Duración onda Q (ms)"] = duracion_q * 1000
    metricas["Profundidad onda Q (mV)"] = profundidad_q
    metricas["Onda Q patológica"] = onda_q_patologica
    metricas["Desviación ST (mV)"] = desviacion_st

    puntos = {
        "indice_r": indice_r,
        "inicio_qrs": inicio_qrs,
        "fin_qrs": fin_qrs,
        "fin_qt": fin_qt,
        "indice_st": indice_st
    }

    return metricas, puntos


# ==========================================================
# ALERTAS Y SEMÁFORO
# ==========================================================

def generar_alertas(metricas, sexo):
    alertas = []

    qrs = metricas["Duración QRS (ms)"]
    qtc = metricas["QTc (ms)"]
    st_dev = metricas["Desviación ST (mV)"]
    onda_q = metricas["Onda Q patológica"]

    if qrs > 120:
        alertas.append("QRS ancho: posible alteración de conducción.")

    if sexo == "Hombre":
        if qtc > 450:
            alertas.append("QTc prolongado en hombre: posible riesgo eléctrico.")
        if st_dev > 0.1:
            alertas.append("Elevación ST en hombre: posible isquemia.")
        if st_dev < -0.1:
            alertas.append("Depresión ST en hombre: posible isquemia.")
    else:
        if qtc > 460:
            alertas.append("QTc prolongado en mujer: posible riesgo eléctrico.")
        if st_dev > 0.15:
            alertas.append("Elevación ST en mujer: posible isquemia.")
        if st_dev < -0.1:
            alertas.append("Depresión ST en mujer: posible isquemia.")

    if onda_q:
        alertas.append("Onda Q patológica: posible infarto previo.")

    if onda_q and (
        (sexo == "Hombre" and st_dev > 0.1) or
        (sexo == "Mujer" and st_dev > 0.15)
    ):
        alertas.append("Patrón sugiere infarto previo o agudo.")

    if len(alertas) == 0:
        semaforo = "Verde"
    elif len(alertas) <= 2:
        semaforo = "Amarillo"
    else:
        semaforo = "Rojo"

    return semaforo, alertas


# ==========================================================
# INTERFAZ PRINCIPAL
# ==========================================================

st.title("Análisis de señales ECG - PTB-XL")

st.write(
    "Panel interactivo para seleccionar pacientes por sexo, edad y condición, "
    "procesar la señal ECG y generar métricas con alertas visuales."
)

st.info(
    "Los metadatos se leen desde el CSV local. La señal ECG (archivo WFDB) "
    "todavía se consulta en PhysioNet; para trabajar completamente sin Internet "
    "también debes tener descargada la carpeta de señales records100."
)

db = cargar_metadatos()
seleccion = seleccionar_pacientes(db)

st.sidebar.header("Selección de paciente")

grupo = st.sidebar.selectbox(
    "Grupo",
    sorted(seleccion["grupo"].unique())
)

df_grupo = seleccion[seleccion["grupo"] == grupo]

paciente_opcion = st.sidebar.selectbox(
    "Paciente / ECG ID",
    df_grupo.index.astype(str).tolist()
)

ecg_id = int(paciente_opcion)
fila = df_grupo.loc[ecg_id]

# Cargar ECG
ecg, fs, derivaciones = leer_ecg_ptbxl(fila["filename_lr"])

derivacion = st.sidebar.selectbox(
    "Derivación ECG",
    derivaciones,
    index=derivaciones.index("II") if "II" in derivaciones else 0
)

canal = derivaciones.index(derivacion)

senal_original = ecg[:, canal]
senal_filtrada = filtro_pasabanda(senal_original, fs)
picos_r = detectar_picos_r(senal_filtrada, fs)

sdnn, rmssd, rr_ms = calcular_hrv(picos_r, fs)

latidos, latido_promedio, tiempo_latido = segmentar_latidos(
    senal_filtrada,
    picos_r,
    fs
)

# ==========================================================
# DATOS DEL PACIENTE
# ==========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Edad", int(fila["age"]))
col2.metric("Sexo", fila["sexo_texto"])
col3.metric("Condición", fila["condicion"])
col4.metric("Frecuencia", f"{fs} Hz")

st.subheader("Información del registro")

st.write({
    "ECG ID": ecg_id,
    "Patient ID": int(fila["patient_id"]),
    "Edad": int(fila["age"]),
    "Sexo": fila["sexo_texto"],
    "Condición": fila["condicion"],
    "Clases diagnósticas": fila["diagnostic_classes"],
    "Reporte clínico": fila["report"],
    "Archivo ECG": fila["filename_lr"]
})


# ==========================================================
# ANÁLISIS
# ==========================================================

if latido_promedio is None:
    st.error("No se pudieron segmentar latidos en este paciente.")
    st.stop()

metricas, puntos = extraer_metricas(
    latido_promedio,
    tiempo_latido,
    fs,
    rr_ms,
    fila["sexo_texto"]
)

metricas["SDNN (ms)"] = sdnn
metricas["RMSSD (ms)"] = rmssd

semaforo, alertas = generar_alertas(metricas, fila["sexo_texto"])

metricas["Semáforo"] = semaforo
metricas["Alertas"] = " | ".join(alertas) if alertas else "Sin alertas relevantes"


# ==========================================================
# SEMÁFORO VISUAL
# ==========================================================

st.subheader("Semáforo automático")

if semaforo == "Verde":
    st.success("🟢 Verde: métricas dentro de rangos esperados.")
elif semaforo == "Amarillo":
    st.warning("🟡 Amarillo: se recomienda monitoreo o revisión.")
else:
    st.error("🔴 Rojo: posible afección detectada.")

if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("Sin alertas relevantes.")


# ==========================================================
# GRÁFICA 1: SEÑAL COMPLETA FILTRADA
# ==========================================================

st.subheader("Señal ECG filtrada con picos R")

tiempo = np.arange(len(senal_filtrada)) / fs

fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=tiempo,
    y=senal_filtrada,
    mode="lines",
    name="ECG filtrado"
))

fig1.add_trace(go.Scatter(
    x=tiempo[picos_r],
    y=senal_filtrada[picos_r],
    mode="markers",
    name="Picos R"
))

fig1.update_layout(
    xaxis_title="Tiempo (s)",
    yaxis_title="Amplitud (mV)",
    height=450
)

st.plotly_chart(fig1, use_container_width=True)


# ==========================================================
# GRÁFICA 2: LATIDO PROMEDIO
# ==========================================================

st.subheader("Latido promedio con ventanas QRS, QT y ST")

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=tiempo_latido,
    y=latido_promedio,
    mode="lines",
    name="Latido promedio"
))

fig2.add_vrect(
    x0=tiempo_latido[puntos["inicio_qrs"]],
    x1=tiempo_latido[puntos["fin_qrs"]],
    fillcolor="orange",
    opacity=0.25,
    line_width=0,
    annotation_text="QRS"
)

fig2.add_vrect(
    x0=tiempo_latido[puntos["inicio_qrs"]],
    x1=tiempo_latido[puntos["fin_qt"]],
    fillcolor="blue",
    opacity=0.15,
    line_width=0,
    annotation_text="QT"
)

fig2.add_vline(
    x=tiempo_latido[puntos["indice_st"]],
    line_dash="dash",
    annotation_text="ST"
)

fig2.update_layout(
    xaxis_title="Tiempo respecto al pico R (s)",
    yaxis_title="Amplitud (mV)",
    height=450
)

st.plotly_chart(fig2, use_container_width=True)


# ==========================================================
# GRÁFICA 3: HISTOGRAMA RR
# ==========================================================

st.subheader("Histograma de intervalos RR")

fig3 = go.Figure()

fig3.add_trace(go.Histogram(
    x=rr_ms,
    nbinsx=15,
    name="Intervalos RR"
))

fig3.update_layout(
    xaxis_title="Intervalo RR (ms)",
    yaxis_title="Frecuencia",
    height=350
)

st.plotly_chart(fig3, use_container_width=True)


# ==========================================================
# TABLA DE MÉTRICAS
# ==========================================================

st.subheader("Tabla de métricas obtenidas")

tabla_metricas = pd.DataFrame({
    "Métrica": list(metricas.keys()),
    "Valor": list(metricas.values())
})

st.dataframe(tabla_metricas, use_container_width=True)


# ==========================================================
# TABLA COMPARATIVA DE RANGOS
# ==========================================================

st.subheader("Comparación contra rangos de referencia usados")

tabla_rangos = pd.DataFrame({
    "Métrica": [
        "Duración QRS",
        "QTc hombres",
        "QTc mujeres",
        "Onda Q patológica",
        "Elevación ST hombres",
        "Elevación ST mujeres",
        "Depresión ST"
    ],
    "Criterio": [
        "<100 ms normal; >120 ms posible anomalía",
        ">450 ms posible riesgo",
        ">460 ms posible riesgo",
        "Duración >0.04 s y profundidad >25% de R",
        ">0.1 mV posible isquemia",
        ">0.15 mV posible isquemia",
        "<-0.1 mV posible isquemia"
    ]
})

st.dataframe(tabla_rangos, use_container_width=True)


# ==========================================================
# RESUMEN DE GRUPOS SELECCIONADOS
# ==========================================================

st.subheader("Pacientes seleccionados por grupo")

resumen = seleccion.groupby("grupo").size().reset_index(name="Cantidad")
st.dataframe(resumen, use_container_width=True)