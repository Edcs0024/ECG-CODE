# Uso con metadatos locales

1. Coloca `ptbxl_dataset_completo.csv` en la misma carpeta que `app_ecg_ptbxl.py` (ya viene incluido en este ZIP).
2. Instala dependencias con `python -m pip install -r requirements.txt`.
3. Ejecuta `python -m streamlit run app_ecg_ptbxl.py`.

La app ya no descarga `ptbxl_database.csv` ni `scp_statements.csv` al iniciar. El CSV incluido contiene metadatos y códigos SCP, pero no las muestras de las señales. Por eso, al seleccionar un registro, `wfdb` todavía intenta leer el ECG desde PhysioNet. Para operación completamente sin Internet, descarga también los archivos WFDB correspondientes (`.dat` y `.hea`) y ajusta la función de lectura para abrirlos desde una carpeta local.

La clasificación `Sano` / `Con afección` es una regla aproximada basada en códigos SCP del CSV, porque no se incluyó `scp_statements.csv` para mapear los códigos a clases diagnósticas. No debe interpretarse como diagnóstico médico.


## Corrección de descarga de señales (v2)

La función `leer_ecg_ptbxl` ahora descarga `.hea` y `.dat` con reintentos,
usa el nombre de registro esperado por WFDB y comprueba que los archivos no
estén vacíos o incompletos. Esto atiende errores como `00002_lr.dat` no encontrado.
Los metadatos continúan cargándose desde `ptbxl_dataset_completo.csv`.
