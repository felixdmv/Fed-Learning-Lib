# Proyecto de Aprendizaje Federado

Este proyecto está diseñado para implementar un sistema de aprendizaje federado utilizando la biblioteca Flower (flwr) y PyTorch. A continuación se describe la estructura del proyecto, el flujo de ejecución y los pasos para ejecutar el sistema.

## Estructura del Proyecto

El proyecto se organiza en varias partes clave:

1. **`main.py`**: Script principal que coordina la ejecución del servidor y los clientes federados.
2. **`start_server.py`**: Inicia el servidor federado con la estrategia de agregación configurada.
3. **`start_clients.py`**: Inicia múltiples instancias de clientes federados, cada uno corriendo el script `fed_client.py`.
4. **`fedavg_strategy.py`**, **`fedprox_strategy.py`**, **`fednova_strategy.py`**: Implementaciones de diferentes estrategias de agregación federada (FedAvg, FedProx, FedNova).
5. **`fed_client.py`**: Implementa la lógica del cliente federado, incluyendo el entrenamiento del modelo y la comunicación con el servidor.
6. **`model.py`**: Define el modelo de red neuronal utilizado para el entrenamiento.
7. **`configuracion.yaml`**: Archivo de configuración que especifica los parámetros del modelo, entrenamiento, servidor, y clientes.

## Flujo de Ejecución

1. **Inicio del Servidor**: 
   - El script `main.py` inicia el servidor federado en segundo plano usando `start_server.py`. El servidor se configura de acuerdo con la estrategia de agregación especificada en `configuracion.yaml`.

2. **Inicio de los Clientes**:
   - Luego de iniciar el servidor, `main.py` lanza el script `start_clients.py` para iniciar múltiples instancias de clientes. Cada cliente se conecta al servidor federado y participa en el proceso de entrenamiento.

3. **Entrenamiento y Evaluación**:
   - Cada cliente realiza el entrenamiento del modelo utilizando los datos particionados. Los clientes envían los parámetros del modelo entrenado al servidor.
   - El servidor agrega los parámetros recibidos de todos los clientes usando la estrategia especificada (FedAvg, FedProx, FedNova) y actualiza el modelo global.
   - Este proceso se repite en varias rondas de entrenamiento según el número de rondas definido en `configuracion.yaml`.

4. **Agregación de Modelos**:
   - En cada ronda, el servidor recibe los parámetros del modelo de todos los clientes, los agrega utilizando la estrategia especificada, y guarda el modelo global. La estrategia de agregación puede incluir métodos como FedAvg (promedio de los parámetros), FedProx (con proximidad), o FedNova (normalización de gradientes).

5. **Finalización del Entrenamiento**:
   - El proceso de entrenamiento y agregación se repite hasta completar el número de rondas definido en `configuracion.yaml`. Tras completar las rondas, el entrenamiento finaliza y se guarda el modelo global.

## Ejecución del Proyecto

Para ejecutar el sistema de aprendizaje federado, sigue estos pasos:

1. **Prepara el Entorno**:
   - Asegúrate de tener todas las dependencias necesarias instaladas (incluyendo `flwr`, `torch`, `numpy`, `pandas`, `pyyaml`).

2. **Configura el Archivo `configuracion.yaml`**:
   - Ajusta los parámetros del modelo, entrenamiento, servidor y clientes según tus necesidades en el archivo `configuracion.yaml`.

3. **Inicia el Sistema**:
   - Ejecuta el script principal con el siguiente comando:

     ```bash
     python main.py
     ```

   - Este comando iniciará el servidor en segundo plano y luego lanzará los clientes federados para comenzar el proceso de entrenamiento federado.

Este proyecto proporciona una base sólida para experimentar con el aprendizaje federado y puede ser extendido con nuevas estrategias, ajustes en el modelo y parámetros de entrenamiento.
