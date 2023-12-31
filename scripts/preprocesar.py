import pandas as pd

_url_original = "data/raw/consumers-price-index-september-2023-quarter-tradables-and-non-tradables.csv"
_url_dataset_limpio = "data/clean/consumers-price-index-september-2023-quarter-tradables-and-non-tradables-clean.csv"
_url_dataset_prediccion = "data/prediccion/predicciones.csv"
_dataset = pd.read_csv(_url_original)
columnas_numericas = _dataset.select_dtypes(include='number')

def __eliminar_outlines(columnas_numericas):
    """Eliminar valores atipicos"""
    menor, mayor = 0.25, 0.75
    quant_col  = columnas_numericas.quantile([menor, mayor])
    columnas_numericas = columnas_numericas.apply(lambda valor: valor[(valor > quant_col.loc[menor, valor.name]) & (valor < quant_col.loc[mayor, valor.name])], axis=0)

def __limpiar_dataset(dataframe):
    """Limpiar el conjunto de datos"""
    valor_mitad = dataframe.count().max() // 2
    dataframe.dropna(axis=1, thresh=valor_mitad, inplace=True)
    dataframe.dropna(inplace=True)
    for columna in columnas_numericas.columns.to_list():
        valor_medio = dataframe[columna].mean()
        dataframe[columna].fillna(valor_medio, inplace=True)
    __eliminar_outlines(columnas_numericas)

def dataset_original_limpio():
    """Devuelve el conjunto de datos limpio"""
    try:
        __limpiar_dataset(_dataset)
        _dataset.to_csv(_url_dataset_limpio, index=False)
    except FileNotFoundError:
        print("El archivo no existe. Puede que haya sido movido o eliminado de la direccion actual. Verifique la direccion")
    else:
        print("Archivo guardado correctamente")
        return pd.read_csv(_url_dataset_limpio)

def guardar_archivo_prediccion(dataset_prediccion, data_value, prediccion):
    """Guardar los resultados de la prediccion"""
    dataset_prediccion = pd.DataFrame({"data_value": data_value, "prediccion": prediccion})
    dataset_prediccion.to_csv(_url_dataset_prediccion, index=False)
    return pd.read_csv(_url_dataset_prediccion)
