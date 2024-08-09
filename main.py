import subprocess
import time

def main():
    # Llamar al script del servidor en segundo plano
    print("Iniciando servidor...")
    server_process = subprocess.Popen(["python", "start_server.py"])
    # Esperar un momento para asegurar que el servidor esté listos
    time.sleep(5)  # Ajusta este tiempo si es necesario
    
    # Llamar al script para iniciar los clientes
    print("Iniciando clientes...")
    client_process = subprocess.Popen(["python", "start_clients.py"])
    # Esperar un momento para asegurar que los clientes estén listos
    time.sleep(5)  # Ajusta este tiempo si es necesario
    
    # Esperar a que el servidor termine si es necesario
    server_process.wait()
    client_process.wait()

if __name__ == "__main__":
    main()  