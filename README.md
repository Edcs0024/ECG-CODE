# Proyecto ECG PTB-XL (versión modificada)

## Cambios
- La aplicación lee `ptbxl_dataset_completo.csv` desde la misma carpeta; ya no descarga los metadatos al iniciar.
- La lectura de códigos `scp_codes` tolera valores vacíos o malformados.
- Los archivos de señal ECG se siguen leyendo de PhysioNet al seleccionar un registro; para eso sí se requiere conexión.
- Se conserva la interfaz y el procesamiento ECG original.

## Ejecutar en Windows
1. Extrae todo el ZIP en una carpeta.
2. Abre PowerShell en esa carpeta (donde está `app_ecg_ptbxl.py`).
3. Instala dependencias: `python -m pip install -r requirements.txt`
4. Ejecuta: `python -m streamlit run app_ecg_ptbxl.py`

El CSV debe permanecer junto a `app_ecg_ptbxl.py`. La clasificación Sano/Con afección es aproximada y no constituye un diagnóstico médico.
