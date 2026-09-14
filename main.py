from settings.settings import settings
from utils.logger import setup_logging
from utils.file_helpers import save_df_to_csv
from modelos.estatico.logistico import initialize
from modelos.dinamico.identificacion import identify
from modelos.features.pipeline import generate_features
from modelos.dominio.customer_builder import CustomerBuilder
from modelos.dominio.customer_director import CustomerDirector
from modelos.evaulacion.backtesting import run_backtesting


setup_logging()

if __name__ in {"__main__", "__mp_main__"}:
    # director = CustomerDirector(CustomerBuilder())
    # save_df_to_csv((df:= director.build_dataset(seed=42)), settings.RAW_TRANSACTIONS_PATH)
    # save_df_to_csv(generate_features(df), settings.FEATURES_PATH)
    # initialize()
    identify(force_train=True)
    # _ = run_backtesting(generate_features(director.build_dataset(seed=39)))