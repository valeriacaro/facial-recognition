# coding=utf-8
import cv2
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.linalg as SL
# Número de famosos en la base de datos
n_famosos = 20
# Número de fotos de cada famoso
mult = 3
# Número de famosos_input (1 foto por famoso)
n_famosos_input = 1
# Número de componentes a utilizar para la compresión
p = 15
#####################################################################################
#########
# obtenerNombresFamosos
#####################################################################################
#########
def obtenerNombresFamosos():
    text = open('famosos.txt')
    #Prelectura de un nombre
    linia = text.readline().rstrip()
    #Inicialitzación de la matriz de carácteres por el char()
    nombres_famosos = []
    #Mientras leemos más carácteres, seguimos guardando datos en la matriz
    while len(linia) > 0:
        nombres_famosos.append(linia)
        linia = text.readline().rstrip()
    return nombres_famosos
#####################################################################################
#########
# obtenerArrayFotos
#####################################################################################
#########
def obtenirArrayFotos(nombreFotos, rutaFotos):
    for i in range (nombreFotos):
        # Abrimos la imagen en Blanco y negro
        dummy = Image.open(rutaFotos + str (i + 1) + '.jpg').convert('L')
        #Reducción de tamaño
        dummy = dummy.resize((75,65), Image.ANTIALIAS)
        #Transformación en columna
        new_column = np.reshape(dummy, (4875, -1))
        if i==0:
            varArrFotos = np.array([new_column.T[0]])
        else:
            varArrFotos = np.append(varArrFotos,np.array([new_column.T[0]]),axis=0)
    varArrFotos = varArrFotos.T
    return varArrFotos


#####################################################################################
#########
# calculaError
#####################################################################################
#########
def calculaError(n_famosos, mult, famosos_p, foto_famos_input):
    errors = np.array([n_famosos * mult, n_famosos * mult])
    norma_errors = []
    for i in range(n_famosos * mult):
        if i==0:
            errors = np.array([(famosos_p[:, i] - foto_famos_input)])
        else:
            errors = np.append(errors,np.array([(famosos_p[:, i] - foto_famos_input)]),axis=0)
        norma_errors.append(np.array([np.linalg.norm(errors[i, :], 2)]))
    #Si tenemos más de una foto por famoso, hacemos la mediana geométrica de las
    #normas de los errores
    if mult > 1:
        norma_final = np.ones(n_famosos)
        for i in range (n_famosos * mult):
            j = math.ceil((i + 1) / mult) - 1
            #print ("i: ", i, " j: ", j)
            norma_final[j] = norma_final[j] * norma_errors[i]
    else:
        norma_final = norma_errors
    #Buscamos la norma mínima
    pos_min = 0
    for i in range (1,n_famosos):
        if norma_final[i] <= norma_final[pos_min]:
            pos_min = i
    return pos_min
#####################################################################################
#########
# INICIO
#####################################################################################
#########
# Leemos el fichero que contiene los nombres de los famosos
nombres_famosos = obtenerNombresFamosos()
# Inicializamos el array de famosos y lo llenamos con las fotos que tenemos
fotos_famosos = obtenirArrayFotos(n_famosos * mult, "./Fotos_Famosos/")
# Inicializamos el array de famosos_input y lo llenamos con las fotos que tenemos
fotos_famosos_input = obtenirArrayFotos(n_famosos_input, "./Fotos_Famosos_Input/")
# Obtenemos la SVD del array de los famosos
U, s , Vtrans = np.linalg.svd(cv2.normalize(fotos_famosos.astype('float'), None, 1.0, 0.0, cv2.NORM_INF), full_matrices = True) # TODO: False o True?
S = np.zeros((fotos_famosos.shape[0],fotos_famosos.shape[1]))
S[:fotos_famosos.shape[1],:fotos_famosos.shape[1]] = np.diag(s)
# Recuperamos la matriz de famosos con la SVD por partes
comprimida = np.matmul(np.matmul(U[:, 0:p], S[0:p , 0:p]), Vtrans[0:p, :])
compression = S.shape
compression = ((compression[0] + compression[1]) * p) / (compression[0] * compression[1])

compression = 100 * (1 - compression)
#####################################################################################
#########
# INICIO RECONOCIMIENTO
#####################################################################################
#########
Utrans = U.T
Utrans = cv2.normalize(Utrans.astype('float'), None, 1.0, 0.0, cv2.NORM_INF)
foto_famos_input = fotos_famosos_input [:, 0]
foto_famos_input = cv2.normalize(foto_famos_input.astype('float'), None, 1.0, 0.0, cv2.NORM_INF)
S_p = S[0:p,0:p]
Vtrans_p = Vtrans[0:p,:]
foto_famos_input = np.matmul(Utrans, foto_famos_input)
foto_famos_input = foto_famos_input [0:p, :]
#Reconstruimos los famosos desde la SVD truncada
famosos_p = np.matmul(S_p, Vtrans_p)
# Calculamos el error y obtenemos la posición mínima donde el error es mínimo
pos_min = calculaError(n_famosos, mult, famosos_p, foto_famos_input)
print("Resultado: ", nombres_famosos[pos_min])
#####################################################################################
#########
# Validación gráfica
#####################################################################################
#########
plt.figure(figsize = (16, 8))
plt.subplot(1, 2, 1)
plt.imshow(np.reshape(fotos_famosos_input [:, 0], [65, 75]),cmap='gray')
plt.axis('off')
plt.title("Famoso Original")

plt.subplot(1, 2, 2)
plt.imshow(np.reshape(fotos_famosos [:, pos_min * mult], [65, 75]),cmap='gray')
plt.axis('off')
plt.title("Resultado")
plt.show()
