# LIBRERÍAS NECESARIAS
import numpy as np
import pandas as pd
import cv2
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from joblib import load
from skimage.morphology import flood_fill

import tkinter as tk
from tkinter import filedialog as fd
from PIL import Image, ImageTk

import serial
import threading

# FUNCIÓN PARA OBTENER LOS DATOS DEL PUERTO
def enviarDatos(pos):
    global estadoHW
    
    with serial.Serial('COM4',9600) as arduino:
        while True:
            try:
                arduino.write(estadoHW.encode('utf-8'))
                print("ENVIANDO DATO: " + estadoHW)
            except:
                print("NO ESTOY FUNCIONANDO BIEN")

def hallarMancha(imagen):
    
    indices = np.where(imagen==255)
    index_list = list(zip(list(indices[0]),list(indices[1])))
    index = index_list[0]
    
    matrizMancha = flood_fill(imagen, index, 0)
    return matrizMancha

def hallarCentro(imgMancha):
    altoImg_sp = 520
    anchoImg_sp = 400
    
    #EROSIÓN + DILATACIÓN DE LA IMAGEN
    kernel = np.ones((5,5),np.uint8)
    imgErosionM = cv2.erode(imgMancha,kernel,iterations=5)
    #imgDilatacionM = cv2.dilate(imgErosionM,kernel,iterations=3)
    
    imgManchaR = cv2.resize(imgErosionM,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
    
    indices = np.where(imgManchaR==255)

    index_list_X = list(indices[1])
    index_list_Y = list(indices[0])
    
    
    if(len(index_list_X) == 0 or len(index_list_Y) == 0):
        centroF = 0
        centroC = 0
    
    else:
        min_X = min(index_list_X)
        max_X = max(index_list_X)
        min_Y = min(index_list_Y)
        max_Y = max(index_list_Y)
        
        centroF = min_Y+int(max_Y-min_Y)/2
        centroC = min_X+int(max_X-min_X)/2
    
    return centroF, centroC

def hallarUbicacion(cF, cC):
    
    if(cC < 200):
        if(cF < 200):
            ubicacion = "Frontal Izquierda"
        elif(cF < 270):
            ubicacion = "Parietal Izquierda"
        elif(cF < 410):
            if(cC < 100):
                ubicacion = "Temporal Izquierda"
            else:
                ubicacion = "Parietal Izquierda"
        else:
            ubicacion = "Occipital Izquierda"
    if(cC >= 200):
        if(cF < 200):
            ubicacion = "Frontal Derecha"
        elif(cF < 270):
            ubicacion = "Parietal Derecha"
        elif(cF < 410):
            if(cC > 300):
                ubicacion = "Temporal Derecha"
            else:
                ubicacion = "Parietal Derecha"
        else:
            ubicacion = "Occipital Derecha"
    if(cF == 0 and cC == 0):
        ubicacion = "Incertidumbre"
    return ubicacion    
    
def importarImagen():
    global imagenImportada
    global imagenIzPI
    global imagenIztk
    global file_path
    global nombre
    global extension
    
    altoImg_sp = 520
    anchoImg_sp = 400
    
    # VENTANA DE BÚSQUEDA DE ARCHIVO A IMPORTAR
    tipoArchivos = [("PNG files", ".png"),("JPG files", ".jpg")]
    direccionInicial = "."
    file_path = fd.askopenfilename(initialdir = direccionInicial, filetypes = tipoArchivos)
    
    # OBTENER NOMBRE DEL ARCHIVO Y SU EXTENSIÓN
    slash = file_path.rfind("/")
    punto = file_path.rfind(".")
    
    nombre = file_path[slash+1:punto]
    extension = file_path[punto:]
    
    # IMPORTAR IMAGEN
    imagenImportada = cv2.imread(file_path)[...,::-1]
    
    # VISUALIZAR IMAGEN IMPORTADA EN LA INTERFAZ CON UN TAMAÑO ADECUADO
    imagenImpor = cv2.resize(imagenImportada,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
    imagenImpor = cv2.cvtColor(imagenImpor, cv2.COLOR_BGR2RGB)
    imagenIzPI = ImageTk.PhotoImage(Image.fromarray(imagenImpor))

    imagenIztk.destroy()

    imagenIztk = tk.Label(frameSuperior, image = imagenIzPI)
    imagenIztk.grid(row = 1, column = 1, padx = 30)

def procesarImagen():
    global tTumor
    global textTumor
    global editTumor
    global tLed
    
    global imagenImportada
    global imagenProcesada
    global modeloPrediccion
    
    global imagenLedConTumor
    global imagenLedSinTumor
    
    global imagenDetk
    global imagenDe
    global imagenDePI
    
    global imagenFL
    global imagenFR
    global imagenPL
    global imagenPR
    global imagenTL
    global imagenTR
    global imagenOL 
    global imagenOR
    global imagenIncer
    
    global estadoHW
    
    altoImg_sp = 520
    anchoImg_sp = 400
        
    
    if 'imagenImportada' in globals():
        img = imagenImportada[:,:,0].copy()
        img.astype(np.uint8)
        
        altura, ancho = img.shape
        
        #UMBRALIZACIÓN DE LA IMAGEN
        _,imgUmb = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    
        #EROSIÓN + DILATACIÓN DE LA IMAGEN
        kernel = np.ones((5,5),np.uint8)
        imgErosion = cv2.erode(imgUmb,kernel,iterations=3)
        imgDilatacion = cv2.dilate(imgErosion,kernel,iterations=3)
    
        #OBTENER PARÁMETROS
        # Porcentaje de Blancos respecto al total de la imagen
        porcBlanco = (np.sum(imgDilatacion))/(altura*ancho)
        
        # Promedio de la intensidad de pixel
        promedio = np.mean(img)
        
        # Desviación Estándar de la intensidad de pixel
        desviacion = np.std(img)
        
        # Máxima intensidad de pixel
        maximo = np.amax(img)
        
        # Mínimo intensidad de pixel
        minimo = np.amin(img)
        
        # Entropia de la imagen
        imgLog = [pixel for pixel in img.ravel() if pixel > 0]
        logEntro = (imgLog*np.log2(imgLog))
        logEntro = logEntro/(100*np.amax(logEntro))
        entropia = np.sum(logEntro)
    
        # Parámetros de CO-OCURRENCIA
        coMatrix = greycomatrix(img, [1], [0]) #DISTANCIA Y DIRECCIÓN
    
        # Correlación
        correlacion = greycoprops(coMatrix, prop='correlation')
          
        # Energía
        energia = greycoprops(coMatrix, prop='energy')
          
        # Homogeneidad
        homogeneidad = greycoprops(coMatrix, prop='homogeneity')
        
        #X = [porcBlanco, promedio, desviacion, maximo, minimo, entropia, correlacion, energia, homogeneidad]
        
        X_test = pd.DataFrame({'pB': [porcBlanco], 'p': [promedio], 'd': [desviacion], 'ma': [maximo], 'mi': [minimo], 'e': [entropia],
                               'c': [correlacion], 'en': [energia], 'h': [homogeneidad]})
        
        Y_pred_prob = modeloPrediccion.predict_proba(X_test)
        prob = Y_pred_prob[:,1]
       
        tLed.destroy()
        imagenDetk.destroy()
       
        if prob > 0.6:
            textTumor.set("CON TUMOR")
            
            # LED CON TUMOR
            tLed = tk.Label(frameInferior, image = imagenLedConTumor, bd = 0, bg = COLORVENTANA, fg = COLORFRTITULO, font = ('Arial',15))
            tLed.grid(row = 2, column = 3, rowspan = 1)
            
            mancha = hallarMancha(imgDilatacion)
            cFila, cColumna = hallarCentro(mancha)
            
            ubicacion = hallarUbicacion(cFila, cColumna)
            
            if(ubicacion == "Frontal Izquierda"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenFL,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "1"
                
            if(ubicacion == "Frontal Derecha"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenFR,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "2"
                
            if(ubicacion == "Parietal Izquierda"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenPL,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "3"
                
            if(ubicacion == "Parietal Derecha"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenPR,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "4"
                
            if(ubicacion == "Temporal Izquierda"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenTL,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "5"
                
            if(ubicacion == "Temporal Derecha"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenTR,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "6"
                
            if(ubicacion == "Occipital Izquierda"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenOL,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "7"
                
            if(ubicacion == "Occipital Derecha"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenOR,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
                
                estadoHW = "8"
                
            if(ubicacion == "Incertidumbre"):
                # IMAGEN PROCESADA
                imagenDeR = cv2.resize(imagenIncer,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
                imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
                imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))
    
                imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
                imagenDetk.grid(row = 1, column = 2, padx = 30)
            
                estadoHW = "9"
            
        else:
            textTumor.set("SIN TUMOR")
            
            tLed = tk.Label(frameInferior, image = imagenLedSinTumor, bd = 0, bg = COLORVENTANA, fg = COLORFRTITULO, font = ('Arial',15))
            tLed.grid(row = 2, column = 3, rowspan = 1)
            
            # IMAGEN PROCESADA
            imagenDeR = cv2.resize(imagenDe,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
            imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
            imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))

            imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
            imagenDetk.grid(row = 1, column = 2, padx = 30)
            
            estadoHW = "0"
        pass
    
    
    else:    
        tk.messagebox.showinfo(message="NO SE HA IMPORTADO NINGUNA IMAGEN", title="Advertencia")
        
        
def exportarImagen():
    pass

############################################
# INICIAR CONSTANTE
############################################
global tTumor
global textTumor
global editTumor
global tLed

global imagenImportada
global imagenIzPI
global imagenIztk
global imagenDe
global imagenDetk
global imagenDePI
global imagenProcesada

global imagenLedConTumor
global imagenLedSinTumor

global imagenFL
global imagenFR
global imagenPL
global imagenPR
global imagenTL
global imagenTR
global imagenOL 
global imagenOR
global imagenIncer

global modeloPrediccion

global posicionHW
global estadoHW

posicionHW = 0
estadoHW = "0"

modeloPrediccion = load("Detector_Tumores.pkl")

TAMANIOVENTANA = '1000x790'

COLORVENTANA = 'gray74'#'dodger blue'
COLORBGTITULO = 'gray21'#'RoyalBlue3'
COLORFGTITULO = "white"
COLORFRTITULO = "black"
FUENTETITULO = ("Arial", 24)
TITULO = "PROYECTO FINAL - DETECCIÓN DE TUMORES - PDI"

ALTOFRAMESUP = 500
ANCHOFRAMESUP = 900
TEXTOFRAMESUP = "Imágenes"

ALTOFRAMEINF = 150
ANCHOFRAMEINF = 900
TEXTOFRAMEINF = "Acciones"

COLORBOTONACT = 'gray74'#'royal blue'
FUENTEBOTON = ("Arial", 12)
FUENTEBOTON2 = ("Arial", 8)

############################################
# CARGAR IMÁGENES DE REFERENCIA INICIALES
############################################
imagenIz = cv2.imread('imagenes/sinImagen.jpg')
imagenDe = cv2.imread('imagenes/lobulos.png')

imagenFL = cv2.imread('imagenes/lobuloFrontalLeft.png')
imagenFR = cv2.imread('imagenes/lobuloFrontalRight.png')
imagenPL = cv2.imread('imagenes/lobuloParietalLeft.png')
imagenPR = cv2.imread('imagenes/lobuloParietalRight.png')
imagenTL = cv2.imread('imagenes/lobuloTemporalLeft.png')
imagenTR = cv2.imread('imagenes/lobuloTemporalRight.png')
imagenOL = cv2.imread('imagenes/lobuloOccipitalLeft.png')
imagenOR = cv2.imread('imagenes/lobuloOccipitalRight.png')
imagenIncer = cv2.imread('imagenes/lobulosIncer.png')

imagenImp = cv2.imread('imagenes/importarCerebro.png')
imagenPro = cv2.imread('imagenes/procesarCerebro.png')
imagenExp = cv2.imread('imagenes/exportarCerebro.png')
imagenApagado = cv2.imread('imagenes/ledApagado.png')
imagenConTumor = cv2.imread('imagenes/ledConTumor.png')
imagenSinTumor = cv2.imread('imagenes/ledSinTumor.png')

altoImg_sp = 520
anchoImg_sp = 400

altoImg_bt = 120
anchoImg_bt = 120

altoImg_sm = 50
anchoImg_sm = 50

############################################
# COMUNICACIÓN
############################################
## HILOS PARALELOS
# Hilo de Obtención de Datos
hilo1_getData = threading.Thread(target = enviarDatos, args = (posicionHW,))   
hilo1_getData.start()


############################################
# CREAR VENTANA PRINCIPAL
############################################

# CREAR VENTANA PRINCIPAL
ventana = tk.Tk()
ventana.title("INTERFAZ DE USUARIO PARA FILTRADO DE IMÁGENES EN EL ESPACIO")
ventana.geometry(TAMANIOVENTANA)
ventana.configure(background = COLORVENTANA)

# CREAR TÍTULO
vTitulo = tk.Label(ventana, text = TITULO, bg = COLORBGTITULO, fg = COLORFGTITULO, font=FUENTETITULO)
vTitulo.pack(fill = tk.X)

# CREAR UN FRAME QUE CONTENDRÁ LAS IMÁGENES
frameSuperior = tk.LabelFrame(ventana, text = TEXTOFRAMESUP, height = ALTOFRAMESUP, width = ANCHOFRAMESUP, bg = COLORVENTANA, fg = COLORFRTITULO, bd = 1)
frameSuperior.pack(pady = 8)

# CREAR UN FRAME QUE CONTENDRÁ LAS CONFIGURACIONES
frameInferior = tk.LabelFrame(ventana, text = TEXTOFRAMEINF, height = ALTOFRAMEINF, width = ANCHOFRAMEINF, bg = COLORVENTANA, fg = COLORFRTITULO, bd = 1)
frameInferior.pack()

############################################
# DISEÑAR FRAME SUPERIOR DE IMÁGENES
############################################

# IMAGEN SIN PROCESAR
imagenIzR = cv2.resize(imagenIz,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
imagenIzPI = ImageTk.PhotoImage(Image.fromarray(imagenIzR))

imagenIztk = tk.Label(frameSuperior, image = imagenIzPI)
imagenIztk.grid(row = 1, column = 1, padx = 30)

# IMAGEN PROCESADA
imagenDeR = cv2.resize(imagenDe,(anchoImg_sp,altoImg_sp),interpolation = cv2.INTER_AREA)
imagenDeR = cv2.cvtColor(imagenDeR, cv2.COLOR_BGR2RGB)
imagenDePI = ImageTk.PhotoImage(Image.fromarray(imagenDeR))

imagenDetk = tk.Label(frameSuperior, image = imagenDePI)
imagenDetk.grid(row = 1, column = 2, padx = 30)

############################################
# DISEÑAR FRAME INFERIOR DE CONFIGURACIONES
############################################

# IMAGEN DEL BOTÓN DE IMPORTAR
imagenImpR = cv2.resize(imagenImp,(anchoImg_bt,altoImg_bt),interpolation = cv2.INTER_AREA)
imagenImpR = cv2.cvtColor(imagenImpR, cv2.COLOR_BGR2RGB)
imagenImpPI = ImageTk.PhotoImage(Image.fromarray(imagenImpR))

imagenImptk = tk.Button(frameInferior, command = importarImagen, text = "IMPORTAR", image = imagenImpPI, compound = tk.TOP, bg = COLORVENTANA, activebackground = COLORBOTONACT, font = FUENTEBOTON)
imagenImptk.grid(row = 1, column = 1, rowspan = 2, padx = 150)

# IMAGEN DEL BOTÓN DE PROCESAR
imagenProR = cv2.resize(imagenPro,(anchoImg_sm,altoImg_sm),interpolation = cv2.INTER_AREA)
imagenProR = cv2.cvtColor(imagenProR, cv2.COLOR_BGR2RGB)
imagenProPI = ImageTk.PhotoImage(Image.fromarray(imagenProR))

imagenProtk = tk.Button(frameInferior, command = procesarImagen, text = "DETECTAR", image = imagenProPI, compound = tk.TOP, bg = COLORVENTANA, activebackground = COLORBOTONACT, font = FUENTEBOTON2)
imagenProtk.grid(row = 1, column = 6, rowspan = 2, padx = 10)

# IMAGEN DEL BOTÓN DE EXPORTAR
#imagenExpR = cv2.resize(imagenExp,(anchoImg_sm,altoImg_sm),interpolation = cv2.INTER_AREA)
#imagenExpR = cv2.cvtColor(imagenExpR, cv2.COLOR_BGR2RGB)
#imagenExpPI = ImageTk.PhotoImage(Image.fromarray(imagenExpR))

#imagenExptk = tk.Button(frameInferior, command = exportarImagen, text = "EXPORTAR", image = imagenExpPI, compound = tk.TOP, bg = COLORVENTANA, activebackground = COLORBOTONACT, font = FUENTEBOTON2)
#imagenExptk.grid(row = 2, column = 6, rowspan = 1, padx = 10)

# COLOCAR TÍTULO DE CAMPO EDITABLE DE VARIABLES DE UMBRAL Y REALCE
tTumor = tk.Label(frameInferior, text = 'Se determinó que el cerebro está:', bd = 0, bg = COLORVENTANA, fg = COLORFRTITULO, font = ('Arial',15), justify='center', anchor=tk.CENTER)
tTumor.grid(row = 1, column = 2, columnspan = 4, padx = 60)
            
# COLOCAR CAMPO EDITABLE DE VARIABLES DE UMBRAL Y REALCE
textTumor = tk.StringVar(frameInferior)
textTumor.set(" ")
            
editTumor = tk.Entry(frameInferior, textvariable = textTumor, state='readonly', justify='center', bg = COLORFGTITULO)
editTumor.config(width = 15, font = ('Arial',12,'bold'))
editTumor.grid(row = 2, column = 4, rowspan = 1)

# LED APAGADO
imagenApagadoPI = cv2.resize(imagenApagado,(anchoImg_sm,altoImg_sm),interpolation = cv2.INTER_AREA)
imagenLedApagado = ImageTk.PhotoImage(Image.fromarray(imagenApagadoPI))

tLed = tk.Label(frameInferior, image = imagenLedApagado, bd = 0, bg = COLORVENTANA, fg = COLORFRTITULO, font = ('Arial',15))
tLed.grid(row = 2, column = 3, rowspan = 1)

# LED CON TUMOR
imagenConTumorPI = cv2.resize(imagenConTumor,(anchoImg_sm,altoImg_sm),interpolation = cv2.INTER_AREA)
imagenConTumorPI = cv2.cvtColor(imagenConTumorPI, cv2.COLOR_BGR2RGB)
imagenLedConTumor = ImageTk.PhotoImage(Image.fromarray(imagenConTumorPI))

# LED SIN TUMOR
imagenSinTumorPI = cv2.resize(imagenSinTumor,(anchoImg_sm,altoImg_sm),interpolation = cv2.INTER_AREA)
imagenSinTumorPI = cv2.cvtColor(imagenSinTumorPI, cv2.COLOR_BGR2RGB)
imagenLedSinTumor = ImageTk.PhotoImage(Image.fromarray(imagenSinTumorPI))

############################################
# BUCLE INFINITO DE LA VENTANA
############################################
# MOSTRAR VENTANA
ventana.mainloop()
print("FINALIZO")
ventana.destroy()
hilo1_getData.join() #ESPERAR HILO