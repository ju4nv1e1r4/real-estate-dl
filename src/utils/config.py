'''
========================== CONSTANTS ==========================

This file contains some constants methods and features to easier your ML developement

Some classes here:
    - Logs
    - Paths
    - MetricsMonitor
'''

import os
import logging

class Logs:
    @staticmethod
    def config_logging(log_file=None, log_level=logging.INFO):
        '''
        Configura o logging e o structlog.
        
        :param log_file: Caminho do arquivo de log (opcional).
        :param log_level: Nível de log (padrão é INFO).
        '''
        import structlog
        from logging import StreamHandler

        handlers = [StreamHandler()]
        if log_file:
            handlers.append(logging.FileHandler(log_file))

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            handlers=handlers
        )

        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

class Paths:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

    @staticmethod
    def ensure_dirs_exist():
        '''
        Cria os diretórios necessários caso eles não existam.
        '''
        for path in [Paths.DATA_DIR, Paths.MODELS_DIR, Paths.LOGS_DIR]:
            os.makedirs(path, exist_ok=True)

class Monitor:
    @staticmethod
    def outliers_analysis(dataframe):
        '''
        Realiza uma análise de outliers, se aplicável.
        '''
        import numpy as np
        import scipy.stats as stats

        z_scores = stats.zscore(dataframe.select_dtypes(include=[np.number]))
        outliers = (abs(z_scores) > 3).sum()
        return outliers


    @staticmethod
    def multicolinearity_analysis(dataframe):
        '''
        Realiza uma análise de multicolinearidade, se aplicável.
        '''
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        vif_data = pd.DataFrame()
        vif_data["feature"] = dataframe.columns
        vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
        return vif_data
