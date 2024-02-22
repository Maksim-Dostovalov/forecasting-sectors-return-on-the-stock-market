import numpy as np
import pandas as pd
import requests
import apimoex
from tqdm import tqdm

request_url = ('https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json')


with requests.Session() as session:
    iss = apimoex.ISSClient(session, request_url)
    stocks = iss.get()

stocks_secids = [x['SECID'] for x in stocks['securities']]
stocks_secnames = [x['SECNAME'] for x in stocks['securities']]

stocks_info = pd.DataFrame(
    {
        'SECID': stocks_secids,
        'SECNAME': stocks_secnames
    }
)

with requests.Session() as session:
    dictionary = {}
    for secid in tqdm(stocks_secids):
        data = apimoex.get_board_history(session, secid, columns=('TRADEDATE', 'CLOSE'))
        dictionary[secid] = data

data_stocks = pd.DataFrame({key: pd.DataFrame(value).set_index('TRADEDATE')['CLOSE'] for key, value in dictionary.items()})

data_stocks.to_csv('./data/data_stocks_moex.csv')
stocks_info.to_csv('./data/stocks_info.csv')

indexes = [
    'MOEXRE', # Строительство
    'MOEXMM', # Металлургия
    'MOEXIT', # Информационные технологии
    'MOEXFN', # Финансы 
    'MOEXEU', # Электроэнергетика
    'MOEXCN', # Потребительский сектор
    'MOEXCH', # Нефтехимическая промышленность
    'MOEXOG', # Нефть и газ
    'MOEXTL', # Телекоммуникации
    'MOEXTN'  # Транспорт
    ]

with requests.Session() as session:
    dictionary = {}
    for secid in tqdm(indexes):
        data = apimoex.get_market_history(
            session, 
            secid,  
            market='index', 
            columns=('TRADEDATE', 'CLOSE')
        )
        dictionary[secid] = data



data_indexes = pd.DataFrame({key: pd.DataFrame(value).set_index('TRADEDATE')['CLOSE'] for key, value in dictionary.items()})

data_indexes.to_csv('./data/data_indexes_moex.csv')
