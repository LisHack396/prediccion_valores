import pandas as pd

_url_original = "data/raw/consumers-price-index-september-2023-quarter-tradables-and-non-tradables.csv"
_dataset = pd.read_csv(_url_original)
columnas_numericas = _dataset.select_dtypes(include='number')

def __eliminar_outlines(dataframe):
    """Eliminar valores atipicos"""
    for columna in dataframe.select_dtypes(include='number').columns:
        Q1 = dataframe[columna].quantile(0.25)
        Q3 = dataframe[columna].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (dataframe[columna] < (Q1 - 1.5 * IQR)) | (dataframe[columna] > (Q3 + 1.5 * IQR))
        return dataframe[~outlier_mask]

def __limpiar_dataset(dataframe):
    """Limpiar el conjunto de datos"""
    dataframe.columns = dataframe.columns.str.lower()
    dataframe.drop_duplicates(subset=['series_reference'], keep="first", inplace=True)
    cols = dataframe.columns[dataframe.isnull().mean() >= 0.5]
    dataframe.drop(columns=cols, inplace=True)
    dataframe.dropna(inplace=True)
    __eliminar_outlines(columnas_numericas)
    dataframe.drop(dataframe[(dataframe['data_value']) < 0].index, axis=0, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    for columna in columnas_numericas.columns.to_list():
        valor_medio = dataframe[columna].mean()
        dataframe[columna].fillna(valor_medio, inplace=True)

def dataset_original_limpio():
    """Devuelve el conjunto de datos limpio"""
    url_dataset_limpio = "data/clean/consumers-price-index-september-2023-quarter-tradables-and-non-tradables-clean.csv"
    try:
        __limpiar_dataset(_dataset)
        _dataset.to_csv(url_dataset_limpio, index=False)
    except FileNotFoundError:
        print("El archivo ha sido movido o no existe. Por favor, revisa la direccion del archivo")
    else:
        print("Archivo guardado correctamente")
        return pd.read_csv(url_dataset_limpio)

def guardar_archivo_prediccion(dataset_prediccion, data_value, prediccion):
    """Guardar los resultados de la prediccion"""
    url_dataset_prediccion = "data/prediccion/predicciones.csv"
    dataset_prediccion = pd.DataFrame({"data_value": data_value, "prediccion": prediccion})
    dataset_prediccion.to_csv(url_dataset_prediccion, index=False)
    return pd.read_csv(url_dataset_prediccion)
