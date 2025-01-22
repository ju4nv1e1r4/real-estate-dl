import pandas as pd
import tensorflow as tf
from colorama import Fore, Style, init
import warnings

warnings.filterwarnings('ignore')

class Preprocessing:
    def __init__(self, dataset_path):
        """
        Class for data pre-processing.

        Args:
            dataset_path (str): Dataset path in CSV format.
        """
        self.dataset_path = dataset_path
        self.dataset = pd.read_csv(dataset_path)

    def report(self):
        init(autoreset=True)

        data = self.dataset
        print(Fore.GREEN + Style.BRIGHT + '=' * 150)
        print(Fore.GREEN + Style.BRIGHT + '============== SOME RELEVANT INFORMATION =============================================================================================================')
        print(Fore.GREEN + Style.BRIGHT + '=' * 150)
        print('\n')
        print(Fore.CYAN + '~> Shape of Dataset: ', data.shape)
        print(Fore.CYAN + '~> Nº Rows:', data.shape[0])
        print(Fore.CYAN + '~> Nº Columns:', data.shape[1])
        print(Fore.CYAN + '\n~> Columns:', data.columns.tolist(), '\n')

        print(Fore.GREEN + Style.BRIGHT + '=' * 150)
        print(Fore.GREEN + Style.BRIGHT + '=================== DTYPES INFORMATION ===============================================================================================================')
        print(Fore.GREEN + Style.BRIGHT + '=' * 150)
        print('\n')
        print(Fore.CYAN + '\n~> String Columns:', data.select_dtypes('object').columns.tolist())
        print(Fore.CYAN + '\n~> Float Columns:', data.select_dtypes('float64').columns.tolist())
        print(Fore.CYAN + '\n~> Integer Columns:', data.select_dtypes('int64').columns.tolist())

    def remove_outliers(self, dtype_selected: str, target: str, columns_to_drop: list):
        """
        Removes outliers from the dataset based on quantiles.

        Args:
            dtype_selected (str): Data type of feature columns (e.g., 'float64', 'int64').
            target (str): Target column name.
            columns_to_drop (str): Columns to be discarded from the feature set.
        """
        init(autoreset=True)

        data = self.dataset
        FEATURE_COLS = data.select_dtypes(dtype_selected).columns.tolist()
        TARGET = target
        
        X = data[FEATURE_COLS].drop(columns=columns_to_drop)
        y = data[TARGET]

        q90 = y.quantile(0.90)
        q95 = y.quantile(0.95)
        q99 = y.quantile(0.99)
        q01 = y.quantile(0.01)
        q05 = y.quantile(0.05)
        q10 = y.quantile(0.10)

        ymin = y.min()
        ymax = y.max()
        ystd = y.std()
        median = y.median()
        ampl = ymax - ymin

        print(Fore.CYAN + Style.BRIGHT + '### BEFORE REMOVE OUTLIERS ###:')
        print(Fore.CYAN + Style.BRIGHT + 'Quantiles:')
        print(Fore.GREEN + 'X (shape): ', X.shape)
        print(Fore.GREEN + 'y (shape): ', y.shape)
        print(Fore.YELLOW + '95% Quantile: {}'.format(q95))
        print(Fore.YELLOW + '99% Quantile: {}'.format(q99))
        print(Fore.YELLOW + '90% Quantile: {}'.format(q90))
        print(Fore.YELLOW + '01% Quantile: {}'.format(q01))
        print(Fore.YELLOW + '05% Quantile: {}'.format(q05))
        print(Fore.YELLOW + '10% Quantile: {}'.format(q10))

        print(Fore.CYAN + Style.BRIGHT + 'Another Stats:')
        print(Fore.YELLOW + 'Min: {}'.format(ymin))
        print(Fore.YELLOW + 'Max: {}'.format(ymax))
        print(Fore.YELLOW + 'Std: {}'.format(ystd))
        print(Fore.YELLOW + 'Median: {}'.format(median))
        print(Fore.YELLOW + 'Amplitude: {}'.format(ampl))

        lower_bound = y.quantile(0.10)  # 10%
        upper_bound = y.quantile(0.90)  # 90%

        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        q90 = y.quantile(0.90)
        q95 = y.quantile(0.95)
        q99 = y.quantile(0.99)
        q01 = y.quantile(0.01)
        q05 = y.quantile(0.05)
        q10 = y.quantile(0.10)

        ymin = y.min()
        ymax = y.max()
        ystd = y.std()
        median = y.median()
        ampl = ymax - ymin

        print(Fore.CYAN + Style.BRIGHT + '### AFTER REMOVE OUTLIERS ###:')

        print(Fore.CYAN + Style.BRIGHT + 'Quantiles:')
        print(Fore.GREEN + 'X (shape): ', X.shape)
        print(Fore.GREEN + 'y (shape): ', y.shape)
        print(Fore.YELLOW + '95% Quantile: {}'.format(q95))
        print(Fore.YELLOW + '99% Quantile: {}'.format(q99))
        print(Fore.YELLOW + '90% Quantile: {}'.format(q90))
        print(Fore.YELLOW + '01% Quantile: {}'.format(q01))
        print(Fore.YELLOW + '05% Quantile: {}'.format(q05))
        print(Fore.YELLOW + '10% Quantile: {}'.format(q10))

        print(Fore.CYAN + Style.BRIGHT + 'Another Stats:')
        print(Fore.YELLOW + 'Min: {}'.format(ymin))
        print(Fore.YELLOW + 'Max: {}'.format(ymax))
        print(Fore.YELLOW + 'Std: {}'.format(ystd))
        print(Fore.YELLOW + 'Median: {}'.format(median))
        print(Fore.YELLOW + 'Amplitude: {}'.format(ampl))

        try:
            X.to_csv('data/processed/train.csv', index=False)
            y.to_csv('data/processed/test.csv', index=False)
            print(Fore.GREEN + 'The datasets were been saved on data/processed with success.')
        except Exception as error:
            print(Fore.RED + 'The datasets could not be saved.')

class Model:
    def __init__(self, X_train):
        '''
        Inicializa a classe com os dados de treino.

        :param X_train: Dados de entrada para treinamento.
        '''
        self.X_train = X_train

    def model(self):
        '''
        Define a arquitetura do modelo de rede neural.

        :return: Objeto do modelo compilado.
        '''
        input_layer = tf.keras.layers.Input(shape=self.X_train.shape[1:])
        h1 = tf.keras.layers.Dense(70, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_layer)
        h2 = tf.keras.layers.Dense(70, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(h1)
        h3 = tf.keras.layers.Dense(70, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(h2)
        h4 = tf.keras.layers.Dense(70, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(h3)
        h5 = tf.keras.layers.Dense(90, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(h4)
        h6 = tf.keras.layers.Dense(90, activation='relu', kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01))(h5)
        concat = tf.keras.layers.concatenate([input_layer, h6])
        output_layer = tf.keras.layers.Dense(1)(concat)

        model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
        model.summary()

        return model
    
    @staticmethod
    def add_checkpoint(path):
        '''
        Cria um callback para salvar o melhor modelo durante o treinamento.

        :param path: Caminho para salvar os pesos do modelo.
        :return: Objeto ModelCheckpoint.
        '''
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        return checkpoint
    
    @staticmethod
    def train(model, X_train, y_train, epochs, checkpoint, X_valid, y_valid, batch_size):
        '''
        Treina o modelo com os dados fornecidos.

        :param model: Modelo compilado do TensorFlow/Keras.
        :param X_train: Dados de entrada para treino.
        :param y_train: Rótulos para treino.
        :param epochs: Número de épocas.
        :param checkpoint: Callback de checkpoint.
        :param X_valid: Dados de validação.
        :param y_valid: Rótulos de validação.
        :param batch_size: Tamanho do lote.
        :return: Histórico do treinamento.
        '''
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            callbacks=[checkpoint],
            validation_data=(X_valid, y_valid),
            batch_size=batch_size
        )
        return history
    
    @staticmethod
    def save_architecture(model, path):
        '''
        Salva a arquitetura do modelo em um arquivo de imagem.

        :param model: Modelo do TensorFlow/Keras.
        :param path: Caminho para salvar a imagem.
        '''
        tf.keras.utils.plot_model(
            model,
            to_file=path,
            show_shapes=True,  # Mostra as formas dos tensores.
            show_layer_names=True,  # Mostra os nomes das camadas.
            expand_nested=False,  # Não expande camadas aninhadas.
            dpi=96  # Ajusta a resolução da imagem.
        )
