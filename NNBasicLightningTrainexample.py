#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023
from nn_libs.nnBasicLightningTrain import BasicLightningTrain
import torch # torch nos permitirá crear tensores.
import torch.nn as nn # torch.nn nos permite crear una red neuronal.
import torch.nn.functional as F # nn.funcional nos da acceso a las funciones de activación y pérdida.
from torch.optim import SGD # optim contiene muchos optimizadores. Aquí, estamos usando SGD, descenso de gradiente estocástico.

import lightning as L # lightning tiene toneladas de herramientas geniales que facilitan las redes neuronales
from torch.utils.data import TensorDataset, DataLoader # estos son necesarios para los datos de entrenamiento

import matplotlib.pyplot as plt ## matplotlib nos permite dibujar gráficos.
import seaborn as sns ## seaborn hace que sea más fácil dibujar gráficos atractivos.

from pytorch_lightning.utilities.seed import seed_everything # esto se agrega para evitar que en diferentes computadoras se
seed_everything(seed=42)                                     # obtengan diferentes resultados.

# Crear las diferentes dosis que queremos correr a través de la red neuronal.
    # torch.linspace() crea la secuencia de números entre 0 y 1 inclusive.
input_doses = torch.linspace(start=0, end=1, steps=11)
#Ahora vamos a graficar la salida de `BasicLightningTrain`, que actualmente no está optimizado, y compararlo con el gráfico que dibujamos anteriormente de la red neuronal optimizada
#Crear la red neuronal.
model = BasicLightningTrain()
    # ahora ejecute las diferentes dosis a través de la red neuronal
output_values = model(input_doses)
    # Ahora dibuje un gráfico que muestre la eficacia de cada dosis.
    # establece el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
    # crea el gráfico  que lo guardemos como PDF
sns.lineplot(x=input_doses,
                y=output_values.detach(), # NOTA: debido a que final_bias tiene un gradiente, llamamos a detach()
                                          #       para devolver un nuevo tensor que solo tiene el valor y no el gradiente.
                color='blue',  #color azul para indicar red neural con gradiente
                linewidth=2.5),
#ahora etiquete los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BasicLightningTrain')
    ## por último, guarde el gráfico como PDF.
plt.savefig('BasicLightningTrain.pdf')
plt.show()
