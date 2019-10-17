#!/bin/python
import scipy as sc
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.signal import freqz
plot = 1
# Generador de señal de ruido
def ruido_gen(t):
    ruido = np.random.randn(np.size(t))*10 # Crea una señal de ruido aleatoria
    ruidoF = (1.0/len(t)*np.fft.fft(ruido)) # Obtiene espectro 
    return (ruido,ruidoF)
# Obtiene las componentes del filtro de paso banda
def bandpass(lowcut, highcut, fs,order=5,ftype='butter',analog=False):
    nyq = 0.5 * fs
    if analog == True:
        wsL = lowcut
        wsH = highcut
    else:
        wsL = lowcut / nyq
        wsH = highcut / nyq
    if ftype == 'butter':
        b, a = signal.butter(order, [wsL, wsH], btype='bandpass',analog=analog)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, 20 ,[wsL, wsH], btype='bandpass',analog=analog)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, 20 ,[wsL, wsH], btype='bandpass',analog=analog)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, .01, 20, [wsL, wsH], btype='bandpass',analog=analog)
    return b, a
# Obtiene las componentes del filtro de supresor de banda
def bandstop(lowcut, highcut, fs, order=5,ftype='butter',analog=False):
    nyq = 0.5 * fs
    if analog == True:
        wsL = lowcut
        wsH = highcut
    else:
        wsL = lowcut / nyq
        wsH = highcut / nyq
    if ftype == 'butter':
        b, a = signal.butter(order, [wsL, wsH], btype='bandstop',analog=analog)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, 20 ,[wsL, wsH], btype='bandstop',analog=analog)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, 20 ,[wsL, wsH], btype='bandstop',analog=analog)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, .01, 20, [wsL, wsH], btype='bandstop',analog=analog)
    return b, a

# Obtiene las componentes del filtro de paso alta
def highpass(highcut,fs,order=5,ftype='butter',analog=False):
    nyq = 0.5 * fs
    if analog == True:
        ws = highcut
    else:
        ws = highcut / nyq
    if ftype == 'butter':
        b, a = signal.butter(order, ws, btype='highpass',analog=analog)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, 20, ws, btype='highpass',analog=analog)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order,20, ws, btype='highpass',analog=analog)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, .01,20, ws, btype='highpass',analog=analog)
    return b, a

# Obtiene las componentes del filtro de paso baja
def lowpass(lowcut,fs,order=5,ftype='butter',analog=False):
    nyq = 0.5 * fs # nyquist
    if analog == True:
        ws = lowcut # Si es analogo se tomara su valor 
    else:
        ws = lowcut / nyq # Si es digital se obtendra un valor que estara entre 0 y 1 
    if ftype == 'butter':
        b, a = signal.butter(order, ws, btype='lowpass',analog=analog)
    elif ftype == 'cheby1':
        b, a = signal.cheby1(order, 20, ws, btype='lowpass',analog=analog)
    elif ftype == 'cheby2':
        b, a = signal.cheby2(order, 20, ws, btype='lowpass',analog=analog)
    elif ftype == 'ellip':
        b, a = signal.ellip(order, .01,20, ws, btype='lowpass',analog=analog)
    return b, a

# Ejecuta la funcion que selecciona el tipo de respuesta del filtro
def filter(fs,data,order=5,lowcut=0,highcut=0,btype='lowpass',ftype='butter',graph=False,analog=False):
    if btype == 'lowpass':
        b, a = lowpass(lowcut, fs, order=order,ftype=ftype,analog=analog)
    elif btype == 'highpass':
        b, a = highpass(highcut, fs, order=order,ftype=ftype,analog=analog)
    elif btype == 'bandpass':
        b, a = bandpass(lowcut, highcut, fs, order=order,ftype=ftype,analog=analog)
    elif btype == 'bandstop':
        b, a = bandstop(lowcut, highcut, fs, order=order,ftype=ftype,analog=analog)
    else:
        return False
    if graph == True:
        return b, a
    else:
        return signal.lfilter(b, a, data)
def label_gen(btype,lowcut,highcut,orders,ordn):
    if btype == "lowpass":
        label = 'Señal pasobajas filtrada a ({0} - {1} Hz), orden de filtrado {2}'.format("inf",lowcut,orders[ordn])
    elif btype == "highpass":
        label = 'Señal pasoaltas filtrada a ({0} - {1} Hz), orden de filtrado {2}'.format(highcut,"inf",orders[ordn])
    elif btype == "bandpass":
        label = 'Señal pasobanda filtrada a ({0} - {1} Hz), orden de filtrado {2}'.format(lowcut,highcut,orders[ordn])
    elif btype == "bandstop":
        label = 'Señal supresora de banda filtrada a ({0} - {1} Hz), orden de filtrado {2}'.format(lowcut,highcut,orders[ordn])
    return label

def run():
    fs = 44100.0 # Define frecuencia de muestreo
    lowcut = 3000.0 # define frecuencia corte bajo
    highcut = 5000.0 # define frecuencia de corte alto
    ftype = 'cheby2' # Aqui se define el tipo de filtro que se usara
    btype = 'highpass' # Aqui se decide que tipo de bandas se dejaran pasar
    T = 0.5 # Se define la longitud en segundos de la señal de ruido
    nsamples = T * fs # Obtiene el numero de muestras
    t = np.linspace(0, T, nsamples, endpoint=False)
    x,f = ruido_gen(t) # Genera señal aleatoria
    
    analog = False
    orders = [1,5,10,15]
    plt.figure(1)
    plt.clf()
    # Itera para graficar las funciones de filtro segun su orden
    for order in orders:
        b, a = filter(fs,x,lowcut=lowcut,highcut=highcut,order=order,btype=btype,ftype=ftype,graph=True,analog=True)
        w, h = signal.freqs(b, a)
        plt.semilogx(w, 20 * np.log10(np.abs(h)), label="Orden = %d" % order)
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Ganancia (dB)')
    plt.grid(True)
    plt.legend(loc='best')

    
    plt.figure(2)
    plt.clf()
    plt.plot(t,0*x-.5)
    plt.plot(t, x - 2.25, label='Señal con ruido')
    n = 100
    yHz = []
    f = np.fft.fftfreq(len(t), 1/fs) 
    # Filtra la señal  aleatoria segun el filtro
    ordn = 0
    for order in orders:
        y = filter(fs,data=x,lowcut=lowcut,highcut=highcut,btype=btype,ftype=ftype,analog=analog)
        yHz.append(np.abs(1.0/len(t)*np.fft.fft(y))) # obtiene muestra de espectro de cada señal
        label = label_gen(btype,lowcut,highcut,orders,ordn)
        plt.plot(t, y - .5 * n, label=label)
        n = n + 100
        ordn = ordn + 1
    plt.xlabel('Tiempo (segundos)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    plt.figure(3)
    plt.clf()
    x = np.abs(1.0/len(t)*np.fft.fft(x))
    plt.plot(f,0*x-.5)
    plt.plot(f, x - 1.00, label='Señal con ruido')
    n = 3
    ordn=0
    for y in yHz:
        label = label_gen(btype,lowcut,highcut,orders,ordn)
        plt.plot(f,y - .5 * n,label=label)
        n = n + 1
        ordn = ordn +1
    plt.xlabel('Frecuencia (Hz)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    run()