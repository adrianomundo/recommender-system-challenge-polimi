from utils.data_handler import *

urm_tuples = data_csv_splitter("urm")
urm_all = urm_all_builder(urm_tuples)

icm_asset = data_csv_splitter("icm_asset")
icm_price = data_csv_splitter("icm_price")
icm_sub_class = data_csv_splitter("icm_sub_class")

icm_all_builder(icm_asset, icm_price, icm_sub_class)
