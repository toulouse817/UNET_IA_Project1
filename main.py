#Asignatura: IA
#Elaborado por: Gabriel Ramírez
#11/02/2023
import subprocess # módulo de subproceso en Python. Obtenemos
# este módulo por defecto cuando se instala Python
#e ste módulo permite ejecutar comandos tales como run(), call(), check_call(), check_output() y otros

def mostrar_menu(opciones):
    print('Seleccione una opción:')
    for clave in sorted(opciones):
        print(f' {clave}) {opciones[clave][0]}')


def leer_opcion(opciones):
    while (a := input('Opción: ')) not in opciones:
        print('Opción incorrecta, vuelva a intentarlo.')
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    opciones = {
        '1': ('NNBasicexample', accion1),
        '2': ('NNBasicTrainexample', accion2),
        '3': ('NNBasicOpexample', accion3),
        '4': ('NNBasicLightningexample', accion4),
        '5': ('NNBasicLightningTrainexample', accion5),
        '6': ('NNBasicLightningOpexample', accion6),
        '7': ('lstmbyhandexample', accion7),
        '8': ('lstmLightningexample', accion8),
        '9': ('Salir', salir)
    }

    generar_menu(opciones, '9')

def accion1():
    print('Has elegido la opción 1')
    print("Ejemplo de Red Neural Básica (NNBasicexample)")
    subprocess.run(["python", "NNBasicexample.py"])
def accion2():
    print('Has elegido la opción 2')
    print("Ejemplo de Red Neural Básica Entrenada (NNBasicTrainexample)")
    subprocess.run(["python", "NNBasicTrainexample.py"])
def accion3():
    print('Has elegido la opción 3')
    print("Ejemplo de Red Neural Básica Optimizada (NNBasicOpexample)")
    subprocess.run(["python", "NNBasicOPexample.py"])
def accion4():
    print('Has elegido la opción 4')
    print("Ejemplo de Red Neural Básica Optimizada (NNBasicLightningexample)")
    subprocess.run(["python", "NNBasicLightningexample.py"])
def accion5():
    print('Has elegido la opción 5')
    print("Ejemplo de Red Neural Básica Entrenada (NNBasicLightningTrainexample)")
    subprocess.run(["python", "NNBasicLightningTrainexample.py"])
def accion6():
    print('Has elegido la opción 6')
    print("Ejemplo de Red Neural Básica con Lightning Optimizada (NNBasicLightningOpexample)")
    subprocess.run(["python", "NNBasicLightningOpexample.py"])
def accion7():
    print('Has elegido la opción 7')
    print("Ejemplo de LSTM hecha a mano (lstmbyhandexample)")
    subprocess.run(["python", "lstmbyhandexample.py"])
def accion8():
    print('Has elegido la opción 3')
    print("Ejemplo de LSTM con Pytorch (lstmLightningexample)")
    subprocess.run(["python", "lstmLightningexample.py"])
def salir():
    print('Adios')

if __name__ == '__main__':
    menu_principal()