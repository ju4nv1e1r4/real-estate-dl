import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.constants import Preprocessing
from src.utils.config import Logs
import warnings

warnings.filterwarnings('ignore')

logger = Logs.config_logging(log_file=None)

preprocessor = Preprocessing('data/external/london_houses_data.csv')
preprocessor.report()

column_to_drop = ['saleEstimate_valueChange.percentageChange',
                  'history_percentageChange',
                  'saleEstimate_valueChange.numericChange',
                  'history_numericChange']

preprocessor.remove_outliers(
    dtype_selected='float64',
    target='history_price',
    columns_to_drop=column_to_drop
)
