# ECG PTB-XL optimizado

## Ejecutar en VS Code (Windows)

En la terminal, dentro de esta carpeta:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_ecg_ptbxl.py
```

La aplicación necesita Internet para consultar los dos CSV de metadatos y descargar solamente el registro ECG seleccionado. Las peticiones tienen tiempos límite y los metadatos quedan en caché durante una hora; el registro seleccionado también queda en caché durante la sesión de Streamlit.

Si la terminal ya tiene una versión anterior ejecutándose, deténla con `Ctrl+C` y ejecuta de nuevo el comando anterior. No se descarga la base completa de PTB-XL.

Nota: las métricas y alertas de esta aplicación son aproximaciones educativas y no sustituyen la interpretación clínica de un profesional.
