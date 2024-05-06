import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

def graficar(dataset_limpio, dataset_prediccion):
    """Graficar el conjunto de datos limpio y el conjunto de datos de la prediccion"""
    plt.style.use('ggplot')
    fig = plt.figure('Prediccion de valores', figsize=(10, 6))
    fig.suptitle("Indice de precios de consumidores en septiembre de 2023", fontsize=15, fontweight='bold')
    
    ax1 = fig.add_subplot(121)
    ax1.set(title="Distribucion de los valores de datos")
    ax2 = fig.add_subplot(122)
    ax2.set(title="Prediccion de los valores de datos", xlim=(0, 25))
    
    sns.kdeplot(data=dataset_limpio['Data_value'], fill=True, color='blue', ax=ax1)
    sns.rugplot(data=dataset_limpio['Data_value'], color='blue', ax=ax1)
    sns.lineplot(data=dataset_prediccion, x="data_value", y="prediccion", ax=ax2)
    
    fig.tight_layout()
    plt.show()