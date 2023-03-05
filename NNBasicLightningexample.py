#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023

#Basado en el script creado por  Joshua Starmer en el tutorial The StatQuest Introducción a la codificación de redes neuronales con PyTorch y Lightning https://youtu.be/khMzi6xPbuM

#NOTA: Este script utiliza La Programación Orientada a Objetos (POO u OOP según sus siglas en inglés) como paradigma de programación que usa objetos y sus interacciones para diseñar aplicaciones y programas
#En el script se pueden encontrar definición de Clases (usando la palabra reservada class), Objetos como instancias  model = BasicNN() y Métodos usando la palabra reservada def
from nn_libs.nnBasicLightning import BasicLightning
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
    # Cree una red neuronal simple en PyTorch
##Al igual que construir una red neuronal ***preentrenada*** en **PyTorch**,
# construir una red neuronal ***preentrenada*** con **PyTorch + Lightning**
# significa crear una nueva clase con dos métodos:
# `__init__()` y `forward()`. El método `__init__()` define e inicializa todos los parámetros que queremos usar,
# y el método `forward()` le dice a **PyTorch + Lightning** lo que debería suceder durante un pase directo a través de la red neuronal.

    #Una vez que hemos creado la clase que define la red neuronal, podemos crear una red neuronal real e imprimir sus parámetros,
    #solo para asegurarnos de que todo es lo que esperamos.
    #Crer la red neuronal.
model = BasicLightning()
    #Imprime el nombre y el valor de cada parámetro
for name, param in model.named_parameters():
    print(name, param.data)
    # Los valores para cada peso y sesgo en `BasicLightning` coinciden con los valores que vemos en la red neuronal optimizada (abajo).

    # Usar la red neuronal y graficar la salida
    # Ahora que tenemos una red neuronal, podemos usarla en una variedad de dosis para determinar cuál será efectiva. Luego, podemos hacer un gráfico de estos datos, y este gráfico debe coincidir con la forma doblada verde que se ajusta a los datos de entrenamiento que
    # se muestran en la parte superior de este documento. Entonces, comencemos por hacer una secuencia de dosis de entrada...

    # Crear las diferentes dosis que queremos correr a través de la red neuronal.
    # torch.linspace() crea la secuencia de números entre 0 y 1 inclusive.
    input_doses = torch.linspace(start=0, end=1, steps=11)
    # ahora imprima las dosis para asegurarse de que son las que esperamos...
    input_doses
    #Crear la red neuronal.
    model = BasicLightning()
    # Ahora ejecute las diferentes dosis a través de la red neuronal.
    output_values = model(input_doses)
# Ahora dibuje un gráfico que muestre la eficacia de cada dosis.
# Primero, configure el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
# cree el gráfico (es posible que no lo vea en este momento, pero lo verá después de que lo guardemos como PDF).
sns.lineplot(x=input_doses,
                         y=output_values,
                         color='red',  #color rojo para indicar red neural básica
                         linewidth=2.5)
# ahora etiquete los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BasicLightning')
plt.savefig('BasicLightning.pdf')
#El gráfico muestra que la red neuronal se ajusta a los datos de entrenamiento.
#guarde el gráfico como PDF.
plt.show()



