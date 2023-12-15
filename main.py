from scripts.preprocesar import dataset_original_limpio
from scripts.predecir import predecir_y_guardar_valores
from scripts.graficar import graficar

if __name__ == "__main__":
    dataset_limpio = dataset_original_limpio().head(50)
    dataset_prediccion = predecir_y_guardar_valores(dataset_limpio).head(50)
    graficar(dataset_limpio, dataset_prediccion)