#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023

#Basado en el script creado por  Joshua Starmer en el tutorial Gran memoria a largo plazo (LSTM) con PyTorch + Lightning https://youtu.be/RHGiXPuo_pI

#NOTA: Este script utiliza La Programación Orientada a Objetos (POO u OOP según sus siglas en inglés) como paradigma de programación que usa objetos y sus interacciones para diseñar aplicaciones y programas
#En el script se pueden encontrar definición de Clases (usando la palabra reservada class), Objetos como instancias  model = BasicNN() y Métodos usando la palabra reservada def

#Importar los módulos que harán todo el trabajo
import torch # torch nos permitirá crear tensores.
import torch.nn as nn # torch.nn nos permite crear una red neuronal.
import torch.nn.functional as F # nn.functional nos da acceso a las funciones de activación y pérdida.
from torch.optim import Adam # optim contiene muchos optimizadores. Esta vez estamos usando Adam
import lightning as L # lightning tiene toneladas de herramientas geniales que facilitan las redes neuronales
from torch.utils.data import TensorDataset, DataLoader # estos son necesarios para los datos de entrenamiento

from pytorch_lightning.utilities.seed import seed_everything
from lstm_libs.lstmbyhand import LSTMbyHand
model = LSTMbyHand()
print("Antes de la optimización, los parámetros son...")
for name, param in model.named_parameters():
        print(name, param.data)
        print("\nAhora comparemos los valores observados y predichos...")
     # NOTA: Para hacer predicciones, pasamos en los primeros 4 días los valores de las acciones
     # en una matriz para cada empresa. En este caso, la única diferencia entre el
     # Los valores de entrada ## para la empresa A y B se producen el primer día. La empresa A tiene 0 y
     # La empresa B tiene 1.
        print("Empresa A: Observada = 0, Predicha =",
          model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
        print("Empresa B: Observada = 1, Predicha =",
          model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
# Con los parámetros no optimizados, el valor predicho para **Empresa A**, **-0.0377**, no es terrible,
# ya que está relativamente cerca del valor observado, **0**. Sin embargo, el valor predicho para **Empresa B**, **-0.0383**,
# _es_ terrible, porque está relativamente lejos del valor observado, **1**. Entonces, eso significa que necesitamos entrenar el LSTM.

#Entrene la unidad LSTM y use Lightning y TensorBoard para evaluar: Parte 1: Primeros pasos"
#Dado que estamos usando entrenamiento **Lightning**, entrenar el LSTM que creamos a mano es bastante fácil.
#Todo lo que tenemos que hacer es crear los datos de entrenamiento y ponerlos en un `DataLoader`..."
#Crear los datos de entrenamiento para la red neuronal.
inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
labels = torch.tensor([0., 1.])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)
#...y luego crea un **Lightning Trainer**, `L.Trainer`, y ajústalo a los datos de entrenamiento.
# **NOTA:** Estamos comenzando con **2000** epochs. Esto puede ser suficiente para optimizar con
# éxito todos los parámetros, pero puede que no.
# Lo averiguaremos después de comparar las predicciones con los valores observados
trainer = L.Trainer(max_epochs=2000)  # con tasa de aprendizaje predeterminada, 0.001 (esta pequeña tasa de aprendizaje
                                      # hace que el aprendizaje sea lento)
trainer.fit(model, train_dataloaders=dataloader)
#Ahora que hemos entrenado el modelo con **2000** epochs, podemos ver cuán buenas son las predicciones..."
print("\nAhora comparemos los valores observados y predichos...")
print("Empresa A: Observada = 0, Predicha =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Empresa B: Observada = 1, Predicha = ", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
##Desafortunadamente, estas predicciones son terribles. Así que tendremos que hacer más entrenamiento.

#Optimizando (Entrenando) los Pesos y Sesgos en el LSTM que hicimos a mano: Parte 2 - Agregar Más Epochs sin Comenzar de Nuevo
#La buena noticia es que debido a que usamos **Lightning**, podemos retomar el entrenamiento donde lo dejamos sin tener que empezar
#de cero. Esto se debe a que cuando entrenamos con **Lightning**, crea archivos _checkpoint_ que realizan un seguimiento de los pesos
#y sesgos a medida que cambian. Como resultado, todo lo que tenemos que hacer para continuar donde lo dejamos es decirle al `Entrenador`
# dónde se encuentran los archivos del punto de control. Esto es increíble y nos ahorrará mucho tiempo ya que no tenemos que volver a entrenar las primeras **2000** épocas. Así que agreguemos **1000** épocas adicionales al entrenamiento".
#Primero, busque dónde se almacenan los archivos de punto de control más recientes
path_to_checkpoint = trainer.checkpoint_callback.best_model_path ## By default, \"best\" = \"most recent\"
print("El nuevo entrenador comenzará donde lo dejó el último, y los datos del punto de control están aquí: " +
          path_to_checkpoint + "\n")
#Luego crea un nuevo Lightning Trainer
trainer = L.Trainer(max_epochs=3000) # Antes, max_epochs=2000, entonces, al establecerlo en 3000,se añadieron  1000 más.
#Y luego llame a fit() usando la ruta a los archivos de punto de control más recientes
#para que podamos continuar donde lo dejamos.
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)
#Ahora que hemos agregado **1000** epochs al entrenamiento, revisemos las predicciones...
print("\nAhora comparemos los valores observados y predichos...")
print("Empresa A: Observado = 0, Predicha =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Empresa B: Observada = 1, Predicha =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
#...las predicciones  son mucho mejores que antes. También podemos consultar los registros con **TensorBoard**
# como parece que aún hay más espacio para mejorar, agreguemos **2000** epochs más al entrenamiento
#Primero, busque dónde se almacenan los archivos de punto de control más recientes
path_to_checkpoint = trainer.checkpoint_callback.best_model_path # Por defecto la ruta es, \"best\" = \"most recent\"
print("The new trainer will start where the last left off, and the check point data is here: " +
          path_to_checkpoint + "\n")
## Then create a new Lightning Trainer
trainer = L.Trainer(max_epochs=5000) ## Before, max_epochs=3000, so, by setting it to 5000, we're adding 2000 more.
## And then call fit() using the path to the most recent checkpoint files
## so that we can pick up where we left off.
trainer.fit(model, train_dataloaders=dataloader, ckpt_path=path_to_checkpoint)
print("Here")
print(path_to_checkpoint)
#Ahora que hemos agregado **2000** epochs más al entrenamiento (para un total de **5000** epochs), revisemos las predicciones...
print("Ahora comparemos los valores observados y predichos...")
print("Empresa A: Observada = 0, Predicha =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
print("Empresa B: Observada = 1, Predicha =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
#La predicción para la **Empresa A** está muy cerca de **0**, que es exactamente lo que queremos,
#y la predicción para la **Empresa B** está cerca de **1**, que es también lo que queremos.
#Por último, imprimamos las estimaciones finales de los pesos y sesgos.
print("After optimization, the parameters are...")
for name, param in model.named_parameters():
        print(name, param.data)
