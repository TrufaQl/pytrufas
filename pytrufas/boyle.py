import os
import zipfile as zp

from pathlib import Path
# ========== ========== ========== ========== ========== #
"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def mainMenu(options):
    extras = ["e", "p"]
    for i in range(len(options)):
        print(f"{i}: {options[i]}")

    while True:
        x = input(">")
        if x in extras: return x

        try:
            y = int(x)
        except:
            print("Error")
            continue
        if y >= 0 and y < len(options):
            return options[y]

""" 
    Solicita valor numerico por consola.
    Args:
        string label: etiqueta para pedir parametro
        int min: valor minimo (inclusive)
        int max: valor maximo permitido (inclusive)
    Returns:
        int value: 
    Raises:
"""
def inputValue(label, min, max):
    while True:
        txt = input(f"{label}: ")
        try:
            value = int(txt)
        except:
            print("Error")
            continue

        if value >= min and value <= max:
            return value

"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def inputParameter(options):
    for i in range(len(options)):
        print(f"{i}: {options[i]}")

    while True:
        x = input(">")
        try:
            y = int(x)
        except:
            print("Error")
            continue
        if y >= 0 and y < len(options):
            return options[y]

"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def freeQuiz(labels):
    answers = []
    for label in labels:
        answer = input(f"{label}: ")
        answers.append(answer)
    return answers

"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def selectFile(path, add=None):
    local = os.listdir(path)
    if add != None:
        local.append(add)

    file = inputParameter(local)
    return file

"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def dir2zip(rootPath, dir):
    dirPath = rootPath / dir
    filePath = rootPath / f"{dir}.zip"

    with zp.ZipFile(filePath, "w", zp.ZIP_LZMA) as zipf:
        for actual, _, files in os.walk(dirPath):
            for file in files:
                relativePath = (Path(actual) / file).relative_to(rootPath)
                globalPath = rootPath / relativePath
                zipf.write(globalPath, relativePath)

"""
    Ingresar Descripcion.
    Args: 
    Returns:
    Raises:
"""
def readzip(zipPath, files):
    datas = []
    for file in files:
        with zp.ZipFile(zipPath) as zipf:
            with zipf.open(file) as file:
                data = file.read()
                datas.append(data)
    return datas

