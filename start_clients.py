import os
import subprocess
from argparse import ArgumentParser
from utils import load_config

# Load configuration
configuracion = load_config('./config/configuracion.yaml')

parser = ArgumentParser(description="Flower Client")

# Configuraciones
num_clients = configuracion['client']['num_clients']
server_address = "127.0.0.1:8080"
save_model_directory = configuracion['model']['save_model_directory']

os.makedirs(os.path.dirname(save_model_directory), exist_ok=True)

processes = []

# Lanzar múltiples clientes
for i in range(num_clients):
    cmd = ["python", "fed_client.py", "--server_address", server_address, "--partition_id", str(i)]
    #cmd = ["python", "customClient.py"]
    p = subprocess.Popen(cmd)
    processes.append(p)

# Esperar a que todos los procesos terminen
for p in processes:
    p.wait()