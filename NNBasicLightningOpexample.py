#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023
from nn_libs.nnBasicLightningTrain import BasicLightningTrain
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
# crear los datos de entrenamiento para la red neuronal.
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
# NOTA: Debido a que tenemos muy pocos datos, es un número irrealmente pequeño
# cantidad de datos, el algoritmo de tasa de aprendizaje, lr_find(), que usamos en la siguiente sección tiene problemas.
# Entonces, el punto aquí es mostrar cómo usar lr_find() cuando tienes una cantidad razonable de datos,
# que falsificamos aquí haciendo 100 copias de las entradas y etiquetas.
inputs = torch.tensor([0., 0.5, 1.] * 100)
labels = torch.tensor([0., 1., 0.] * 100)
# Si queremos usar Lightning para el entrenamiento, entonces tenemos que pasarle al Entrenador los datos envueltos en
# algo llamado DataLoader. DataLoaders proporciona un puñado de características agradables que incluyen...
# 1) Pueden acceder a los datos en minilotes en lugar de todos a la vez. En otras palabras,
#    El DataLoader no necesita que carguemos todos los datos en la memoria primero. En cambio
#    simplemente carga lo que necesita de manera eficiente. Esto es crucial para grandes conjuntos de datos.
# 2) Pueden reorganizar los datos cada epoch para reducir el sobreajuste del modelo
# 3) Podemos usar fácilmente una fracción de los datos si queremos hacer un entrenamiento rápido
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)
#Y ahora que tenemos algunos datos de entrenamiento, lo primero que debemos hacer es encontrar la **tasa de aprendizaje** óptima para
# el descenso de gradiente.
# Para ello, creamos un **Lightning** `Trainer` y lo usamos para llamar a `tuner.lr_find()` para encontrar una tasa de aprendizaje mejorada.
model = BasicLightningTrain() # Primero, se hace un modelo de la clase.
# Ahora cree un Entrenador: podemos usar el entrenador para...
# 1) Encuentra la tasa de aprendizaje óptima
# 2) Entrenar (optimizar) los pesos y sesgos en el modelo
# De forma predeterminada, el entrenador se ejecutará en la CPU de su sistema
trainer = L.Trainer(max_epochs=34)

## Ahora encontremos la tasa de aprendizaje óptima
lr_find_results = trainer.tuner.lr_find(model,
                                        train_dataloaders=dataloader, # the training data
                                        min_lr=0.001, # tasa mínima de aprendizaje
                                        max_lr=1.0,   # tasa máxima de aprendizaje
                                        early_stop_threshold=None) # establecer esto en "None" prueba las 100 tasas
new_lr = lr_find_results.suggestion() # suggestion() devuelve la mejor suposición para la tasa de aprendizaje óptima
# ahora imprime la tasa de aprendizaje
print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
## ahora establezca la tasa de aprendizaje del modelo en el nuevo valor
model.learning_rate = new_lr
#Ahora que tenemos una tasa de entrenamiento mejorada, entrenemos el modelo para optimizar `final_bias`
#Ahora que tenemos una tasa de aprendizaje mejorada, podemos entrenar el modelo (optimizar final_bias)
trainer.fit(model, train_dataloaders=dataloader)
print(model.final_bias.data)

# Crear las diferentes dosis que queremos correr a través de la red neuronal.
# torch.linspace() crea la secuencia de números entre 0 y 1 inclusive.
input_doses = torch.linspace(start=0, end=1, steps=11)
# ejecutar las diferentes dosis a través de la red neuronal
output_values = model(input_doses)
# establezca el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
# cree el gráfico que lo guardemos como PDF).
sns.lineplot(x=input_doses,
    y=output_values.detach(), # NOTA: llamamos a detach() porque final_bias tiene un gradiente
    color='green', #color green para indicar red neural con optimización en final_bias
    linewidth=2.5)
# ahora etiquete los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BasicLightningTrain_optimized')
# por último, guarde el gráfico como PDF.
plt.savefig('BasicLightningTrain_optimized.pdf')
plt.show()




