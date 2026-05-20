import wfdb
import matplotlib.pyplot as plt
import numpy as np

record_name = "00001_lr"
database = "ptb-xl/1.0.3/records100/00000"

print("Registro:", record_name)
print("Base:", database)

record = wfdb.rdrecord(record_name, pn_dir=database)

ecg = record.p_signal
fs = record.fs
leads = record.sig_name

print("Señal cargada correctamente")
print("Tamaño:", ecg.shape)
print("Frecuencia:", fs)
print("Derivaciones:", leads)

canal = leads.index("II") if "II" in leads else 0
t = np.arange(ecg.shape[0]) / fs

plt.figure(figsize=(12, 4))
plt.plot(t, ecg[:, canal])
plt.title(f"ECG PTB-XL - Derivación {leads[canal]}")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud (mV)")
plt.grid()
plt.show()