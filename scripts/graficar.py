import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

def graficar(dataset_limpio, dataset_prediccion):
    plt.style.use('ggplot')
    fig = plt.figure('Prediccion de valores', figsize=(10, 6))
    ax1, ax2 = plt.subplots(ncols=2)
    fig.suptitle("Indice de precios de consumidores en septiembre de 2023", fontsize=15, fontweight='bold')
    
    ax1.set(title="Distribucion de los valores")
    ax2.set(title="Predicion de valores", xlim=(0, 20))
    
    sns.kdeplot(data=dataset_limpio['Data_value'], fill=True, color='blue', ax=ax1)
    sns.rugplot(data=dataset_limpio['Data_value'], color='blue', ax=ax1)
    sns.lineplot(data=dataset_prediccion, x="data_value", y="prediccion", ax=ax2)
    
    fig.tight_layout()
    plt.show()