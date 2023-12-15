from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scripts.preprocesar import guardar_archivo_prediccion, columnas_numericas

def predecir_y_guardar_valores(dataframe):
    correlacion = __analizar_correlacion()
    if correlacion < 0.7:
        X_set = dataframe.drop('Data_value', axis='columns')
        y_set = dataframe['Data_value']
        X_train, X_test, y_train, y_test = train_test_split(X_set, y_set, test_size=0.2, random_state=1234, shuffle=True)
        preprocesador = __preprocesar_datos(X_train)
        modelo = Pipeline(steps=[('preprocesado', preprocesador), ('modelo', LinearRegression())])
        modelo.fit(X_train, y_train)
        prediccion = modelo.predict(X_test)
        print("Resultados del modelo:")
        print(f"Scoring: {modelo.score(X_test, y_test)}")
        print(f"Error del test: {mean_squared_error(y_true=y_test, y_pred=prediccion, squared=False)}")
        dataframe_prediccion = guardar_archivo_prediccion(dataframe, y_test, prediccion)
        return dataframe_prediccion
    else:
        raise Exception("No se puede llevar a cabo la regresion lineal. Las variables estan altamente correlacionadas")

def __analizar_correlacion():
    correlacion = columnas_numericas.corr(method='pearson').stack().reset_index()
    correlacion.columns = ['variable_1', 'variable_2', 'r']
    correlacion = correlacion.loc[correlacion['variable_1'] != correlacion['variable_2'], :]
    correlacion['abs_r'] = abs(correlacion['r'])
    correlacion = correlacion.sort_values('abs_r', ascending=False)
    correlacion = correlacion.loc[correlacion['variable_1'] == 'Data_value', 'abs_r']
    return correlacion.loc[2]

def __preprocesar_datos(X_train):
    columnas_numericas = X_train.select_dtypes(include=['number']).columns.to_list()
    columnas_categoricas = X_train.select_dtypes(include=['object']).columns.to_list()
    transformacion_numerica = Pipeline(steps=[('scaler', StandardScaler())])
    transformacion_categorica = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    tupla_numerica = ('numerica', transformacion_numerica, columnas_numericas)
    tupla_categorica = ('categoria', transformacion_categorica, columnas_categoricas)
    transformers = [tupla_numerica, tupla_categorica]
    preprocesador = ColumnTransformer(transformers=transformers, remainder='passthrough', verbose_feature_names_out=False)
    return preprocesador.set_output(transform='pandas')
