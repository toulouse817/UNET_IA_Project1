#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023
from nn_libs.nnBasicTrain import BasicNN_train
from nn_libs.nnBasic import BasicNN
# Importación de módulos con funcionalidades especiales para crear m graficar y optimizar redes neurales en Python
import torch  # torch es el nombre a usar para el modulo de PyTorch, proporciona funciones básicas, desde establecer
# una semilla aleatoria (para reproducibilidad) hasta crear tensores.
import torch.nn as nn  # torch.nn  permite crear una red neuronal.
import torch.nn.functional as F  # nn.functional nos da acceso a las funciones de activación y pérdida.
from torch.optim import SGD  # optim contiene muchos optimizadores. Se usará SGD, descenso de gradiente estocástico.
import matplotlib.pyplot as plt  ## matplotlib nos permite dibujar gráficos.
import seaborn as sns  ## seaborn hace que sea más fácil dibujar gráficos atractivos.
from nn_libs.nnBasicTrain import BasicNN_train
# crear la red neuronal.
model = BasicNN_train()
input_doses = torch.linspace(start=0, end=1, steps=11)
# ahora se ejecuta las diferentes dosis a través de la red neuronal.
output_values = model(input_doses)
# Ahora dibuje un gráfico que muestre la eficacia de cada dosis.
# se establezca el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
# Se crea el gráfico que se verá después de que lo guardemos como PDF
sns.lineplot(x=input_doses,
             y=output_values.detach(), # NOTA: debido a que final_bias tiene un gradiente, llamamos a detach()
                                       # para devolver un nuevo tensor que solo tiene el valor y no el gradiente.
             color='blue', #Gráfica configurada en azul para red neural con entrenamiento
             linewidth=2.5)
#ahora se etiqueta los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BasicNN_train.')
# por último, guarde el gráfico como PDF.
plt.savefig('BasicNN_train.pdf')
plt.show()  #Muestra la gráfica
