# ============================================================
# ANÁLISIS DE SEÑALES ECG
# Filtrado, detección de picos R, segmentación, métricas,
# alertas y gráficas.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# 1. CARGA DE SEÑAL
# ============================================================

def cargar_ecg_csv(ruta_csv, columna_ecg="ecg", columna_tiempo=None, fs=500):
    """
    Carga una señal ECG desde un archivo CSV.

    Parámetros:
    ruta_csv: ruta del archivo CSV.
    columna_ecg: nombre de la columna donde está la señal ECG.
    columna_tiempo: nombre de la columna de tiempo, si existe.
    fs: frecuencia de muestreo si no hay columna de tiempo.

    Retorna:
    tiempo, señal, fs
    """

    datos = pd.read_csv(ruta_csv)

    ecg = datos[columna_ecg].values

    if columna_tiempo is not None:
        tiempo = datos[columna_tiempo].values
        fs = 1 / np.mean(np.diff(tiempo))
    else:
        tiempo = np.arange(len(ecg)) / fs

    return tiempo, ecg, fs


def cargar_ecg_wfdb(ruta_registro, canal=0):
    """
    Carga una señal ECG en formato WFDB.

    Ejemplo de ruta:
    "mitdb/100" o "datos/paciente01"

    Requiere instalar:
    pip install wfdb
    """

    import wfdb

    registro = wfdb.rdrecord(ruta_registro)
    ecg = registro.p_signal[:, canal]
    fs = registro.fs
    tiempo = np.arange(len(ecg)) / fs

    return tiempo, ecg, fs


# ============================================================
# 2. FILTRADO PASA BANDA 0.5 - 40 Hz
# ============================================================

def filtro_pasabanda(ecg, fs, lowcut=0.5, highcut=40, orden=4):
    """
    Aplica filtro pasa banda para eliminar ruido de línea base
    y ruido muscular.
    """

    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(orden, [low, high], btype="band")
    ecg_filtrado = filtfilt(b, a, ecg)

    return ecg_filtrado


# ============================================================
# 3. DETECCIÓN DE PICOS R
# ============================================================

def detectar_picos_r(ecg_filtrado, fs):
    """
    Detecta picos R de la señal ECG filtrada.
    """

    distancia_minima = int(0.4 * fs)
    altura_minima = np.mean(ecg_filtrado) + 0.5 * np.std(ecg_filtrado)

    picos, propiedades = find_peaks(
        ecg_filtrado,
        distance=distancia_minima,
        height=altura_minima
    )

    return picos


# ============================================================
# 4. SEGMENTACIÓN DE LATIDOS
# ============================================================

def segmentar_latidos(ecg_filtrado, picos_r, fs, antes=0.25, despues=0.45):
    """
    Segmenta latidos individuales alrededor de cada pico R.
    """

    muestras_antes = int(antes * fs)
    muestras_despues = int(despues * fs)

    latidos = []

    for r in picos_r:
        inicio = r - muestras_antes
        fin = r + muestras_despues

        if inicio >= 0 and fin < len(ecg_filtrado):
            latidos.append(ecg_filtrado[inicio:fin])

    latidos = np.array(latidos)

    if len(latidos) == 0:
        return None, None

    latido_promedio = np.mean(latidos, axis=0)
    tiempo_latido = np.linspace(-antes, despues, len(latido_promedio))

    return tiempo_latido, latido_promedio


# ============================================================
# 5. CÁLCULO DE HRV: SDNN Y RMSSD
# ============================================================

def calcular_hrv(picos_r, fs):
    """
    Calcula variabilidad de frecuencia cardíaca.
    """

    if len(picos_r) < 3:
        return np.nan, np.nan, np.array([])

    rr = np.diff(picos_r) / fs
    rr_ms = rr * 1000

    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))

    return sdnn, rmssd, rr_ms


# ============================================================
# 6. EXTRACCIÓN DE MÉTRICAS ECG
# ============================================================

def extraer_metricas(latido_promedio, tiempo_latido, fs, rr_ms, sexo="Hombre"):
    """
    Extrae métricas aproximadas:
    - Duración QRS
    - QT
    - QTc
    - Amplitud R
    - Onda Q patológica
    - Segmento ST
    """

    metricas = {}

    # Pico R
    indice_r = np.argmax(latido_promedio)
    amplitud_r = latido_promedio[indice_r]

    # Umbral para estimar QRS
    umbral = 0.15 * np.max(np.abs(latido_promedio))

    izquierda = indice_r
    while izquierda > 0 and abs(latido_promedio[izquierda]) > umbral:
        izquierda -= 1

    derecha = indice_r
    while derecha < len(latido_promedio) - 1 and abs(latido_promedio[derecha]) > umbral:
        derecha += 1

    duracion_qrs = (derecha - izquierda) / fs

    # QT aproximado
    fin_qt = min(len(latido_promedio) - 1, indice_r + int(0.35 * fs))
    intervalo_qt = (fin_qt - izquierda) / fs

    # QT corregido con fórmula de Bazett
    if len(rr_ms) > 0:
        rr_promedio = np.mean(rr_ms) / 1000
        qtc = intervalo_qt / np.sqrt(rr_promedio)
    else:
        qtc = np.nan

    # Onda Q aproximada
    inicio_q = max(0, indice_r - int(0.08 * fs))
    segmento_q = latido_promedio[inicio_q:indice_r]

    if len(segmento_q) > 0:
        q_min = np.min(segmento_q)
        indice_q = np.argmin(segmento_q) + inicio_q

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
    punto_j = derecha
    indice_st = min(len(latido_promedio) - 1, punto_j + int(0.06 * fs))

    zona_base = latido_promedio[max(0, izquierda - int(0.08 * fs)):izquierda]

    if len(zona_base) > 0:
        linea_base = np.mean(zona_base)
    else:
        linea_base = 0

    desviacion_st = latido_promedio[indice_st] - linea_base

    # Guardar métricas
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
        "inicio_qrs": izquierda,
        "fin_qrs": derecha,
        "fin_qt": fin_qt,
        "indice_st": indice_st
    }

    return metricas, puntos


# ============================================================
# 7. SISTEMA DE ALERTAS
# ============================================================

def generar_alertas(metricas, sexo="Hombre"):
    """
    Genera alertas visuales tipo semáforo.
    """

    alertas = []

    qrs = metricas["Duración QRS (ms)"]
    qtc = metricas["QTc (ms)"]
    st = metricas["Desviación ST (mV)"]
    onda_q = metricas["Onda Q patológica"]

    if qrs > 120:
        alertas.append("QRS ancho: posible alteración de conducción.")

    if sexo == "Hombre":
        if qtc > 450:
            alertas.append("QTc prolongado en hombre: posible riesgo eléctrico.")
        if st > 0.1:
            alertas.append("Elevación del ST en hombre: posible isquemia o infarto.")
    else:
        if qtc > 460:
            alertas.append("QTc prolongado en mujer: posible riesgo eléctrico.")
        if st > 0.15:
            alertas.append("Elevación del ST en mujer: posible isquemia o infarto.")

    if onda_q:
        alertas.append("Onda Q patológica: posible infarto previo.")

    if onda_q and (
        (sexo == "Hombre" and st > 0.1) or
        (sexo == "Mujer" and st > 0.15)
    ):
        alertas.append("Patrón sugiere infarto previo o agudo.")

    if len(alertas) == 0:
        semaforo = "Verde"
    elif len(alertas) <= 2:
        semaforo = "Amarillo"
    else:
        semaforo = "Rojo"

    return semaforo, alertas


# ============================================================
# 8. GRÁFICAS
# ============================================================

def graficar_senal_completa(tiempo, ecg_filtrado, picos_r):
    """
    Grafica la señal ECG completa filtrada con picos R.
    """

    plt.figure(figsize=(14, 5))
    plt.plot(tiempo, ecg_filtrado, label="ECG filtrado")
    plt.scatter(tiempo[picos_r], ecg_filtrado[picos_r], color="red", label="Picos R")

    plt.title("Señal ECG filtrada con detección de picos R")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (mV)")
    plt.legend()
    plt.grid()
    plt.show()


def graficar_latido_promedio(tiempo_latido, latido_promedio, puntos):
    """
    Grafica latido promedio con zonas QRS, QT y ST.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(tiempo_latido, latido_promedio, label="Latido promedio")

    plt.axvspan(
        tiempo_latido[puntos["inicio_qrs"]],
        tiempo_latido[puntos["fin_qrs"]],
        alpha=0.3,
        label="QRS"
    )

    plt.axvspan(
        tiempo_latido[puntos["inicio_qrs"]],
        tiempo_latido[puntos["fin_qt"]],
        alpha=0.2,
        label="QT"
    )

    plt.axvline(
        tiempo_latido[puntos["indice_st"]],
        linestyle="--",
        label="Punto ST"
    )

    plt.title("Latido promedio con regiones QRS, QT y ST")
    plt.xlabel("Tiempo respecto al pico R (s)")
    plt.ylabel("Amplitud (mV)")
    plt.legend()
    plt.grid()
    plt.show()


def graficar_histograma_rr(rr_ms):
    """
    Grafica histograma de intervalos RR.
    """

    plt.figure(figsize=(8, 5))
    plt.hist(rr_ms, bins=15, edgecolor="black")

    plt.title("Histograma de intervalos RR")
    plt.xlabel("Intervalo RR (ms)")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.show()


# ============================================================
# 9. ANÁLISIS DE UN PACIENTE
# ============================================================

def analizar_paciente(
    ruta,
    tipo_archivo="csv",
    columna_ecg="ecg",
    columna_tiempo=None,
    fs=500,
    canal=0,
    sexo="Hombre",
    edad=30,
    condicion="Sano"
):
    """
    Función principal para analizar un paciente.
    """

    print("\n==============================================")
    print("ANÁLISIS DE PACIENTE ECG")
    print("==============================================")
    print(f"Edad: {edad}")
    print(f"Sexo: {sexo}")
    print(f"Condición registrada: {condicion}")

    # Cargar señal
    if tipo_archivo == "csv":
        tiempo, ecg, fs = cargar_ecg_csv(
            ruta_csv=ruta,
            columna_ecg=columna_ecg,
            columna_tiempo=columna_tiempo,
            fs=fs
        )

    elif tipo_archivo == "wfdb":
        tiempo, ecg, fs = cargar_ecg_wfdb(
            ruta_registro=ruta,
            canal=canal
        )

    else:
        raise ValueError("tipo_archivo debe ser 'csv' o 'wfdb'.")

    # Filtrado
    ecg_filtrado = filtro_pasabanda(ecg, fs)

    # Picos R
    picos_r = detectar_picos_r(ecg_filtrado, fs)

    # HRV
    sdnn, rmssd, rr_ms = calcular_hrv(picos_r, fs)

    # Latido promedio
    tiempo_latido, latido_promedio = segmentar_latidos(ecg_filtrado, picos_r, fs)

    if latido_promedio is None:
        print("No se pudieron segmentar latidos.")
        return None

    # Métricas
    metricas, puntos = extraer_metricas(
        latido_promedio,
        tiempo_latido,
        fs,
        rr_ms,
        sexo=sexo
    )

    metricas["SDNN (ms)"] = sdnn
    metricas["RMSSD (ms)"] = rmssd

    # Alertas
    semaforo, alertas = generar_alertas(metricas, sexo)

    metricas["Semáforo"] = semaforo
    metricas["Alertas"] = " | ".join(alertas) if alertas else "Sin alertas relevantes"

    # Mostrar resultados
    print("\nMÉTRICAS OBTENIDAS")
    print("----------------------------------------------")

    for clave, valor in metricas.items():
        print(f"{clave}: {valor}")

    print("\nALERTAS")
    print("----------------------------------------------")

    if alertas:
        for alerta in alertas:
            print(f"- {alerta}")
    else:
        print("- Sin alertas relevantes.")

    print(f"\nSEMAFORO FINAL: {semaforo}")

    # Gráficas
    graficar_senal_completa(tiempo, ecg_filtrado, picos_r)
    graficar_latido_promedio(tiempo_latido, latido_promedio, puntos)

    if len(rr_ms) > 0:
        graficar_histograma_rr(rr_ms)

    return metricas


# ============================================================
# 10. EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":
    print("Módulo de análisis ECG cargado correctamente.")
    print("Para analizar una señal, llama a analizar_paciente(...) con la ruta real del archivo.")
    # Ejemplo:
    # analizar_paciente(
    #     ruta="paciente01.csv",
    #     tipo_archivo="csv",
    #     columna_ecg="ecg",
    #     columna_tiempo=None,
    #     fs=500,
    #     sexo="Hombre",
    #     edad=35,
    #     condicion="Sano",
    # )
