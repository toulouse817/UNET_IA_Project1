#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023
from nn_libs.nnBasic import BasicNN
# Importación de módulos con funcionalidades especiales para crear m graficar y optimizar redes neurales en Python
import torch  # torch es el nombre a usar para el modulo de PyTorch, proporciona funciones básicas, desde establecer
# una semilla aleatoria (para reproducibilidad) hasta crear tensores.
import torch.nn as nn  # torch.nn  permite crear una red neuronal.
import torch.nn.functional as F  # nn.functional nos da acceso a las funciones de activación y pérdida.
from torch.optim import SGD  # optim contiene muchos optimizadores. Se usará SGD, descenso de gradiente estocástico.
import matplotlib.pyplot as plt  ## matplotlib nos permite dibujar gráficos.
import seaborn as sns  ## seaborn hace que sea más fácil dibujar gráficos atractivos.

# Crear la red neuronal.
model = BasicNN()
# Imprimir el nombre y el valor de cada parámetro
for name, param in model.named_parameters():
    print(name, param.data)
input_doses = torch.linspace(start=0, end=1, steps=11)
# ahora se imprime las dosis para asegurarse de que son las que esperamos...
input_doses
# Ahora que tenemos `input_doses`, vamos a ejecutarlas a través de la red neuronal y graficar la salida...
## crear la red neuronal.
model = BasicNN()
# Ahora se ejecuta las diferentes dosis a través de la red neuronal.
output_values = model(input_doses)
# Ahora se dibuja un gráfico que muestre la eficacia de cada dosis.
# Primero, se configure el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
# Se crea el gráfico que lo verá después de que lo guardemos como PDF
sns.lineplot(x=input_doses,
             y=output_values,
             color='red',  # Gráfica configurada en rojo para red neural sin optimizar
             linewidth=2.5)
##Se etiqueta los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BasicNN')
# Se guarda el gráfico como PDF.
plt.savefig('BasicNN.pdf')
plt.show()  # Muestra la gráfica

