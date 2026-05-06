from metaflow import Flow

def main():
    # Obtener la última ejecución exitosa de nuestro pipeline
    run = Flow('ChurnPredictionFlow').latest_successful_run
    
    print(f"--- Datos recuperados de la ejecución: {run.id} ---")
    
    # Metaflow guarda automáticamente todo lo que asignamos a 'self' en el pipeline
    # bajo 'run.data'. Vamos a recuperar el mejor modelo.
    best_model_name = run.data.best_model_name
    best_score = run.data.best_score
    best_model = run.data.best_model
    
    print(f"Mejor Modelo: {best_model_name}")
    print(f"F1-Score: {best_score:.4f}")
    
    # ¡Aquí tienes tu modelo real de scikit-learn listo para usarse!
    print(f"Objeto del modelo: {type(best_model)}")
    
    # Para recuperar datos que se guardaron ANTES de un branch paralelo (@foreach),
    # debemos pedírselos específicamente al paso donde se crearon (paso 'start'):
    X_test = run['start'].task.data.X_test
    print(f"Ejemplo: Recuperados los datos de prueba con forma {X_test.shape}")

if __name__ == '__main__':
    main()
