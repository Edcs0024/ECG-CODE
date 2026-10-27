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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


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
# MÉTRICAS DE VARIOS REGISTROS PARA K-MEANS
# ==========================================================

@st.cache_data(show_spinner=False, max_entries=300)
def metricas_registro_para_cluster(filename_lr, sexo):
    """Calcula las mismas métricas ECG para un registro, para alimentar K-Means."""
    ecg, fs, derivaciones = leer_ecg_ptbxl(filename_lr)
    canal = derivaciones.index("II") if "II" in derivaciones else 0
    senal = ecg[:, canal]
    filtrada = filtro_pasabanda(senal, fs)
    picos = detectar_picos_r(filtrada, fs)
    sdnn, rmssd, rr_ms = calcular_hrv(picos, fs)
    latidos, promedio, tiempo = segmentar_latidos(filtrada, picos, fs)
    if promedio is None:
        raise ValueError("No se pudieron segmentar latidos")
    metricas, _ = extraer_metricas(promedio, tiempo, fs, rr_ms, sexo)
    metricas["SDNN (ms)"] = sdnn
    metricas["RMSSD (ms)"] = rmssd
    return metricas


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

st.sidebar.header("Selección de paciente")
st.sidebar.caption("Escribe el Patient ID que quieras consultar.")

patient_ids = sorted(db["patient_id"].dropna().astype(int).unique().tolist())
patient_id_min = int(min(patient_ids))
patient_id_max = int(max(patient_ids))

patient_id_elegido = st.sidebar.number_input(
    "Número de paciente (Patient ID)",
    min_value=patient_id_min,
    max_value=patient_id_max,
    value=patient_id_min,
    step=1,
    help="Usa el Patient ID del conjunto PTB-XL; no es lo mismo que el ECG ID."
)

filas_paciente = db[db["patient_id"].astype(int) == int(patient_id_elegido)]

if filas_paciente.empty:
    st.warning(
        f"El Patient ID {int(patient_id_elegido)} no aparece en los registros "
        "disponibles. Prueba con otro número dentro del rango mostrado."
    )
    st.stop()

registros_paciente = sorted(filas_paciente["ecg_id"].astype(int).unique().tolist())
if len(registros_paciente) > 1:
    ecg_id = st.sidebar.selectbox(
        "Registro ECG de este paciente",
        registros_paciente,
        help="Un paciente puede tener más de un registro ECG."
    )
else:
    ecg_id = registros_paciente[0]
    st.sidebar.caption(f"Registro ECG disponible: {ecg_id}")

fila = db.loc[ecg_id]

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
# SEMÁFORO COMBINADO: REGLAS + CONTEXTO K-MEANS
# ==========================================================

# Se conservan las reglas actuales. K-Means solo puede elevar Verde a Amarillo
# si detecta un patrón estadísticamente atípico; nunca reduce alertas.
resultados_kmeans_guardados = st.session_state.get("kmeans_resultados")
fila_cluster_actual = None
if isinstance(resultados_kmeans_guardados, pd.DataFrame) and not resultados_kmeans_guardados.empty:
    coincidencia = resultados_kmeans_guardados[
        resultados_kmeans_guardados["ECG ID"].astype(int) == int(ecg_id)
    ]
    if not coincidencia.empty:
        fila_cluster_actual = coincidencia.iloc[0]

semaforo_combinado = semaforo
if (
    fila_cluster_actual is not None
    and bool(fila_cluster_actual.get("Patrón estadísticamente atípico", False))
    and semaforo_combinado == "Verde"
):
    semaforo_combinado = "Amarillo"

st.subheader("Semáforo automático combinado")

if semaforo_combinado == "Verde":
    st.success("🟢 Verde: no hay alertas por reglas y K-Means no marcó este registro como atípico en la muestra analizada.")
elif semaforo_combinado == "Amarillo":
    st.warning("🟡 Amarillo: revisar las métricas o el patrón estadístico del registro.")
else:
    st.error("🔴 Rojo: se activaron tres o más alertas basadas en reglas.")

st.caption(
    f"Semáforo por reglas: {semaforo}. K-Means aporta contexto estadístico, no un diagnóstico. "
    "Si todavía no ejecutaste K-Means o el registro no está en la muestra, se muestra únicamente el resultado por reglas."
)

if fila_cluster_actual is not None:
    st.info(
        f"Registro ECG {int(ecg_id)}: clúster K-Means "
        f"{int(fila_cluster_actual['Clúster K-Means'])}. "
        f"Patrón estadísticamente atípico: "
        f"{'Sí' if bool(fila_cluster_actual['Patrón estadísticamente atípico']) else 'No'}."
    )

if alertas:
    for alerta in alertas:
        st.warning(alerta)
else:
    st.info("Sin alertas activadas por las reglas actuales.")


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

seleccion = seleccionar_pacientes(db)
st.subheader("Pacientes seleccionados por grupo")
resumen = seleccion.groupby("grupo").size().reset_index(name="Cantidad")
st.dataframe(resumen, use_container_width=True)


# ==========================================================
# K-MEANS: AGRUPAMIENTO NO SUPERVISADO DE VARIOS ECG
# ==========================================================

st.header("Agrupamiento no supervisado con K-Means")
st.write(
    "K-Means agrupa registros por similitud entre sus métricas ECG. "
    "Los grupos son patrones matemáticos y no representan diagnósticos médicos."
)

col_k1, col_k2 = st.columns(2)
with col_k1:
    cantidad_registros = st.slider(
        "Cantidad de registros para analizar", min_value=12, max_value=100, value=30, step=6,
        help="Cada registro requiere descargar/procesar la señal si todavía no está en caché."
    )
with col_k2:
    numero_clusters = st.selectbox("Número de clústeres (K)", [2, 3, 4, 5], index=1)

if st.button("Ejecutar K-Means", type="primary"):
    # Incluir siempre el ECG seleccionado y completar con otros registros.
    universo = db.reset_index(drop=True).drop_duplicates("ecg_id")
    registro_actual = universo[universo["ecg_id"].astype(int) == int(ecg_id)].head(1)
    restantes = universo[universo["ecg_id"].astype(int) != int(ecg_id)]
    cantidad_restantes = min(max(cantidad_registros - len(registro_actual), 0), len(restantes))
    restantes = (
        restantes.sample(n=cantidad_restantes, random_state=42)
        if cantidad_restantes > 0 else restantes.head(0)
    )
    muestra = pd.concat([registro_actual, restantes], ignore_index=True).drop_duplicates("ecg_id")
    filas_metricas = []
    errores = []
    barra = st.progress(0, text="Preparando registros ECG...")
    for i, (_, registro) in enumerate(muestra.iterrows(), start=1):
        try:
            valores = metricas_registro_para_cluster(
                str(registro["filename_lr"]), str(registro["sexo_texto"])
            )
            fila_resultado = {
                "ECG ID": int(registro["ecg_id"]),
                "Patient ID": int(registro["patient_id"]),
                "Edad": int(registro["age"]),
                "Sexo": registro["sexo_texto"],
                "Condición dataset": registro["condicion"],
            }
            fila_resultado.update(valores)
            filas_metricas.append(fila_resultado)
        except Exception as exc:
            errores.append(f"ECG {registro.get('ecg_id', '?')}: {exc}")
        barra.progress(i / len(muestra), text=f"Procesando registro {i} de {len(muestra)}...")
    barra.empty()

    if len(filas_metricas) <= numero_clusters:
        st.error(
            f"Solo se pudieron calcular métricas para {len(filas_metricas)} registros. "
            f"Se necesitan más de {numero_clusters} para formar y evaluar los clústeres. "
            "Revisa la conexión a PhysioNet o intenta de nuevo."
        )
    else:
        resultados = pd.DataFrame(filas_metricas)
        columnas_modelo = [
            "Duración QRS (ms)", "Intervalo QT (ms)", "QTc (ms)",
            "Amplitud R (mV)", "Duración onda Q (ms)",
            "Profundidad onda Q (mV)", "Desviación ST (mV)",
            "SDNN (ms)", "RMSSD (ms)"
        ]
        X = resultados[columnas_modelo].replace([np.inf, -np.inf], np.nan)
        # Evitar imputar datos inventados: se excluyen registros con métricas incompletas.
        mascara_valida = X.notna().all(axis=1)
        resultados_validos = resultados.loc[mascara_valida].copy()
        X = X.loc[mascara_valida]

        if len(X) <= numero_clusters:
            st.error(
                "No hay suficientes registros con todas las métricas válidas para este K. "
                "Aumenta la cantidad de registros o reduce el número de clústeres."
            )
        else:
            escalador = StandardScaler()
            X_escalado = escalador.fit_transform(X)
            modelo = KMeans(n_clusters=numero_clusters, random_state=42, n_init=10)
            etiquetas = modelo.fit_predict(X_escalado)
            resultados_validos["Clúster K-Means"] = etiquetas

            # Distancia a centroide; atípico significa superar el percentil 95
            # de las distancias de esta muestra, no una condición médica.
            distancias = np.linalg.norm(
                X_escalado - modelo.cluster_centers_[etiquetas], axis=1
            )
            umbral_atipico = float(np.percentile(distancias, 95))
            resultados_validos["Distancia al centroide"] = distancias
            resultados_validos["Patrón estadísticamente atípico"] = distancias > umbral_atipico
            st.session_state["kmeans_resultados"] = resultados_validos.copy()
            st.session_state["kmeans_umbral_atipico"] = umbral_atipico
            st.session_state["kmeans_numero_clusters"] = numero_clusters

            grupos_presentes = len(np.unique(etiquetas))
            if grupos_presentes > 1 and grupos_presentes < len(etiquetas):
                silueta = silhouette_score(X_escalado, etiquetas)
                st.metric("Silhouette score", f"{silueta:.3f}")
                st.caption(
                    "El silhouette score va aproximadamente de -1 a 1; valores mayores suelen "
                    "indicar grupos mejor separados. Interprétalo como una medida matemática, no clínica."
                )
            else:
                st.info("No se pudo calcular silhouette score porque el agrupamiento produjo una sola etiqueta o grupos no evaluables.")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Cantidad de registros por clúster")
                conteos = resultados_validos["Clúster K-Means"].value_counts().sort_index()
                st.bar_chart(conteos)
            with c2:
                st.subheader("Proyección visual de los grupos")
                if X_escalado.shape[1] >= 2:
                    componentes = PCA(n_components=2, random_state=42).fit_transform(X_escalado)
                    grafica = pd.DataFrame({
                        "Componente 1": componentes[:, 0],
                        "Componente 2": componentes[:, 1],
                        "Clúster": etiquetas.astype(str),
                    })
                    fig_cluster = go.Figure()
                    for etiqueta in sorted(grafica["Clúster"].unique()):
                        sub = grafica[grafica["Clúster"] == etiqueta]
                        fig_cluster.add_trace(go.Scatter(
                            x=sub["Componente 1"], y=sub["Componente 2"],
                            mode="markers", name=f"Clúster {etiqueta}",
                            text=[f"Clúster {etiqueta}" for _ in range(len(sub))],
                        ))
                    fig_cluster.update_layout(
                        xaxis_title="Componente principal 1",
                        yaxis_title="Componente principal 2", height=420
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)

            st.subheader("Semáforo combinado del ECG seleccionado")
            coincidencia_actual = resultados_validos[
                resultados_validos["ECG ID"].astype(int) == int(ecg_id)
            ]
            if not coincidencia_actual.empty:
                datos_actuales = coincidencia_actual.iloc[0]
                semaforo_final = semaforo
                es_atipico_actual = bool(datos_actuales["Patrón estadísticamente atípico"])
                if es_atipico_actual and semaforo_final == "Verde":
                    semaforo_final = "Amarillo"

                if semaforo_final == "Verde":
                    st.success("🟢 Verde: sin alertas por reglas ni atipicidad estadística detectada en esta muestra.")
                elif semaforo_final == "Amarillo":
                    st.warning("🟡 Amarillo: hay alertas por reglas o el registro es estadísticamente atípico.")
                else:
                    st.error("🔴 Rojo: tres o más alertas por reglas; K-Means no modifica este estado.")

                st.write({
                    "ECG ID seleccionado": int(ecg_id),
                    "Clúster K-Means": int(datos_actuales["Clúster K-Means"]),
                    "Distancia al centroide": round(float(datos_actuales["Distancia al centroide"]), 3),
                    "Umbral estadístico (percentil 95)": round(umbral_atipico, 3),
                    "Patrón estadísticamente atípico": "Sí" if es_atipico_actual else "No",
                    "Semáforo por reglas": semaforo,
                    "Semáforo combinado": semaforo_final
                })
            else:
                st.info("El ECG seleccionado no tuvo métricas válidas y no se pudo combinar su clúster con el semáforo.")

            st.subheader("Registros y clúster asignado")
            columnas_resultado = [
                "ECG ID", "Patient ID", "Edad", "Sexo", "Condición dataset",
                "Clúster K-Means", "Distancia al centroide",
                "Patrón estadísticamente atípico"
            ] + columnas_modelo
            st.dataframe(resultados_validos[columnas_resultado], use_container_width=True)
            st.download_button(
                "Descargar resultados K-Means (CSV)",
                data=resultados_validos.to_csv(index=False).encode("utf-8-sig"),
                file_name="resultados_kmeans_ecg.csv",
                mime="text/csv"
            )
            if errores:
                with st.expander(f"Registros omitidos por error ({len(errores)})"):
                    st.write("Algunos registros no pudieron procesarse. No se incluyeron en el agrupamiento.")
                    st.code("\n".join(errores[:30]))
            st.warning(
                "Los clústeres son identificadores arbitrarios. El indicador atípico se basa en la "
                "distancia al centroide respecto al percentil 95 de esta muestra; no equivale a riesgo "
                "clínico ni a un diagnóstico. El rojo depende únicamente de las reglas existentes."
            )