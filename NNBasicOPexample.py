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
# comencemos por hacer una secuencia de dosis de entrada...
# Ahora crea las diferentes dosis que queremos que corran a través de la red neuronal.
# torch.linspace() crea la secuencia de números entre 0 y 1
input_doses = torch.linspace(start=0, end=1, steps=11)
# Ahora que tenemos `input_doses`, vamos a ejecutarlas a través de la red neuronal y graficar la salida...
## crear la red neuronal.
model = BasicNN()
#Ahora se ejecuta las diferentes dosis a través de la red neuronal.
output_values = model(input_doses)
# crear los datos de entrenamiento para la red neuronal.
inputs = torch.tensor([0., 0.5, 1.])
labels = torch.tensor([0., 1., 0.])
# ...y ahora usemos esos datos de entrenamiento para entrenar (u optimizar) `final_bias`.
# crear la red neuronal que queremos entrenar.
# ...y ahora usemos esos datos de entrenamiento para entrenar (u optimizar) `final_bias`.
# crear la red neuronal que queremos entrenar.
model = BasicNN_train()

optimizer = SGD(model.parameters(), lr=0.1) ## here we're creating an optimizer to train the neural network.
                                            ## NOTA: Hay muchas formas diferentes de optimizar una red neuronal.
                                            ## En este ejemplo, usaremos el descenso de gradiente estocástico (SGD).
print("Final bias, before optimization: " + str(model.final_bias.data) + "\n")
## este es el bucle de optimización. Cada vez que el optimizador ve todos los datos de entrenamiento se denomina "epoch".
for epoch in range(100):
    # creamos e inicializamos total_loss para cada epoch para  evaluar qué tan bien se ajusta el modelo al
    # datos de entrenamiento. Al principio, cuando el modelo no se ajusta muy bien a los datos de entrenamiento, total_loss
    # será grande. Sin embargo, a medida que el descenso del gradiente mejora el ajuste, la pérdida total se hará cada vez más pequeña.
    # Si la pérdida total se vuelve muy pequeña, podemos decidir que el modelo se ajusta lo suficientemente bien a los datos y detenernos
    # de optimizar el ajuste. De lo contrario, podemos seguir optimizando hasta alcanzar el número máximo de epochs.
    total_loss = 0
    ## este ciclo interno es donde el optimizador ve todos los datos de entrenamiento y donde
    ## calcule la pérdida total para todos los datos de entrenamiento.
    for iteration in range(len(inputs)):
        input_i = inputs[iteration] #extrae un solo valor de entrada (una sola dosis)...
        label_i = labels[iteration] #y su etiqueta correspondiente (la eficacia para la dosis).
        output_i = model(input_i) #calcula la salida de la red neuronal para la entrada (la dosis única).
        loss = (output_i - label_i)**2  # calcula la pérdida para el valor único.
                                        # NOTA: Debido a que output_i = model(input_i), "loss" tiene una conexión con "model"
                                        # y la derivada (calculada en el siguiente paso) se mantiene y se acumula
                                        # en "model".
        loss.backward() # backward() calcula la derivada de ese único valor y la suma a la anterior.
        total_loss += float(loss) # acumula la pérdida total por este epoch.
    if (total_loss < 0.0001):
        print("Num steps: " + str(epoch))
        break
    optimizer.step() # dar un paso hacia el valor óptimo.
    optimizer.zero_grad()   # Esto pone a cero el gradiente almacenado en "model".
                            # Recuerde, por defecto, los gradientes se agregan al paso anterior (los gradientes se acumulan),
                            # y aprovechamos este proceso para calcular la derivada de un punto de datos a la vez.
                            # NOTA: "optimizer" tiene acceso a "model" debido a cómo se creó con la llamada
                            # (hecho antes): optimizer = SGD(model.parameters(), lr=0.1).
                            # TAMBIÉN NOTA: Alternativamente, podemos poner a cero el gradiente con model.zero_grad().
    print("Step: " + str(epoch) + " Final Bias: " + str(model.final_bias.data) + "\n")
    # ahora vuelve al inicio del bucle y pasa por otra epoch.
print("Total loss: " + str(total_loss))
print("Final bias, after optimization: " + str(model.final_bias.data))
# Entonces, si todo funcionó correctamente, el optimizador debería haber convergido en `final_bias = 16.0019`
# después de **34** pasos o épocas. **BAM!**
# Por último, representemos gráficamente la salida de la red neuronal optimizada y veamos si es igual a la que
# comenzamos. Si es así, entonces la optimización funcionó.

# ejecutar las diferentes dosis a través de la red neuronal
output_values = model(input_doses)
#Se establece el estilo para seaborn para que el gráfico se vea bien.
sns.set(style="whitegrid")
#Se crea el gráfico que lo guardemos como PDF).
sns.lineplot(x=input_doses,
             y=output_values.detach(), ## NOTE: we call detach() because final_bias has a gradient
             color='green', #color específico configurado para gráfica optimizada con gradiente
             linewidth=2.5)
# ahora se etiqueta los ejes y y x.
plt.ylabel('Eficacia')
plt.xlabel('Dosis')
plt.suptitle('BascNN_optimized.')
## por último, guarde el gráfico como PDF.
plt.savefig('BascNN_optimized.pdf')
plt.show()  #Muestra la gráfica
# Y vemos que el modelo optimizado da como resultado el mismo gráfico con el que comenzamos, por lo que
# la optimización funcionó como se esperaba.
