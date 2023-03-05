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
from lstm_libs.lstmLightning import LightningLSTM
#Ahora vamos a crear el modelo e imprimir los pesos y sesgos iniciales y las predicciones.
model = LightningLSTM() # Primero, haz un modelo de la clase.
## imprime el nombre y el valor de cada parámetro
print("Antes de la optimización, los parámetros son....")
for name, param in model.named_parameters():
    print(name, param.data)
    print("Ahora comparemos los valores observados y predichos...")
    print("Empresa A: Observado = 0, Predicho =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Empresa B: Observada = 1, Predicha =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
    ## Como era de esperar, las predicciones son malas, por lo que entrenaremos el modelo. Sin embargo,
    # debido a que hemos aumentado la tasa de aprendizaje a **0.1**, solo necesitamos entrenar durante **300** epochs".
    ## NOTA: Debido a que hemos establecido la tasa de aprendizaje de Adam en 0.1, entrenaremos mucho, mucho más rápido.
    ## Antes, con el LSTM hecho a mano y la tasa de aprendizaje predeterminada, 0.001, tomaba alrededor de 5000 epochs
    # para entrenar completamente
    ## el modelo. Ahora, con la tasa de aprendizaje establecida en 0,1, solo necesitamos 300 epochs. Ahora,
    # porque estamos haciendo tan pocas epochs
    # tenemos que decirle al entrenador que agregue cosas a los archivos de registro cada 2 pasos
    # (o epoch, ya que tenemos filas de datos de entrenamiento)
    ## porque el valor predeterminado, actualizar los archivos de registro cada 50 pasos, dará como resultado gráficos
    # de aspecto terrible. Entonces
    # Crear los datos de entrenamiento para la red neuronal.
    inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset)
    trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)
    trainer.fit(model, train_dataloaders=dataloader)
    print("Después de la optimización, los parámetros son...")
    for name, param in model.named_parameters():
        print(name, param.data)
##Ahora que el entrenamiento ha terminado, imprimamos las nuevas predicciones..."
    print("Ahora comparemos los valores observados y predichos...")
    print("Empresa A: Observada = 0, Predicha =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Empresa B: Observada = 1, Predicha =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
#...y, como podemos ver, después de solo **300** epochs, el LSTM está haciendo grandes predicciones.
# La predicción para la **Empresa A** está cerca del valor observado **0** y la predicción para la **Empresa B**
# está cerca del valor observado **1**.
#Por último, actualicemos la página de **TensorBoard** para ver los gráficos más recientes.

