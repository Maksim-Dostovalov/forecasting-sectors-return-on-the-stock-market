# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

set_random_seed(42)

import xgboost as xgb
from catboost import CatBoostRegressor

from typing import List, Union, Tuple

def MASE(y_true: pd.Series, y_pred: pd.Series) -> float:
    '''
    Вычисляет Mean Absolute Scaled Error (MASE) между фактическими и прогнозируемыми значениями.

    Параметры:
            y_true (pd.Series): Серия с фактическими значениями.
            y_pred (pd.Series): Серия с прогнозируемыми значениями.

    Возвращаемое значение:
            float: Значение MASE.
    '''
    # Создание наивного прогноза с использованием сдвига временного ряда
    naive_forecast = y_true.shift(1)
    naive_forecast.iloc[0] = y_true.iloc[0]

    # Вычисление средней абсолютной ошибки для модели и наивного прогноза
    average_errors = np.abs(y_true - y_pred)
    naive_average_errors = np.abs(y_true - naive_forecast)

    result = np.mean(average_errors) / np.mean(naive_average_errors)
    return result


def ZBMAE(y_true: pd.Series, y_pred: pd.Series) -> float:
    '''
    Вычисляет Zero-Benchmarked Mean Absolute Error (ZBMAE) между фактическими и прогнозируемыми значениями.

    ZBMAE сравнивает среднюю абсолютную ошибку прогноза с средней абсолютной ошибкой наивного прогноза, который предполагает, что все прогнозируемые значения равны нулю. Это может быть полезно для временных рядов, где ожидается, что значения будут вокруг нуля.

    Параметры:
        y_true (pd.Series): Серия с фактическими значениями.
        y_pred (pd.Series): Серия с прогнозируемыми значениями.

    Возвращаемое значение:
        float: Значение ZBMAE.
    '''
    zero_forecast = 0

    average_errors = np.abs(y_true - y_pred)
    naive_average_errors = np.abs(y_true - zero_forecast)

    result = np.mean(average_errors) / np.mean(naive_average_errors.mean())
    return result

def SMAPE(y_true: pd.Series, y_pred: pd.Series) -> float:
    '''
    Вычисляет Symmetric Mean Absolute Percentage Error (SMAPE) между фактическими и прогнозируемыми значениями.

    Параметры:
            y_true (pd.Series): Серия с фактическими значениями.
            y_pred (pd.Series): Серия с прогнозируемыми значениями.

    Возвращаемое значение:
            float: Значение SMAPE.
    '''
    # Избегание деления на ноль добавлением небольшого числа в знаменатель
    denominator = (np.abs(y_true) + np.abs(y_pred) + np.finfo(float).eps) / 2
    result = (100 / len(y_true)) * np.sum(np.abs(y_pred - y_true) / denominator)
    return result

def MDA(y_true: pd.Series, y_pred: pd.Series) -> float:

    '''
    Вычисляет Mean Directional Accuracy (MDA) между фактическими и прогнозируемыми значениями.

    MDA измеряет процент времени, когда прогноз и фактическое значение имеют одинаковое направление изменений (например, оба увеличиваются или оба уменьшаются).

    Параметры:
            y_true (pd.Series): Серия с фактическими значениями.
            y_pred (pd.Series): Серия с прогнозируемыми значениями.

    Возвращаемое значение:
            float: Значение MDA в процентах.
    '''

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    actual_diff = np.diff(y_true)
    actual_signs = np.sign(actual_diff)
    predicted_diff = np.diff(y_pred)
    predicted_signs = np.sign(predicted_diff)
    
    num_correct = np.sum(actual_signs == predicted_signs)
    
    mda = num_correct / (len(y_true) - 1)
    return 100*mda


def DMDA(y_true: pd.Series, y_pred: pd.Series) -> float:
    '''
    Вычисляет Direct Mean Directional Accuracy (DMDA) между фактическими и прогнозируемыми значениями.
    

    Параметры:
            y_true (pd.Series): Серия с фактическими значениями.
            y_pred (pd.Series): Серия с прогнозируемыми значениями.

    Возвращаемое значение:
            float: Значение MDA в процентах.
    '''
    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)

    result = (100 / len(y_true)) * np.sum(y_pred_sign == y_true_sign)
    return result

class TimeSeriesProcessor:
    '''
    Класс для работы с временными рядами, включая добавление лагов, интеграцию внешних переменных,
    создание разбиений для валидации, вычисление метрик и визуализацию результатов.

    Методы:
        add_lags(self, lags: List[int], drop_na: bool = True):
            Добавление лагов к переменным временного ряда.
        add_other_variables(self, other_variables: pd.DataFrame):
            Добавление дополнительных переменных к данным.
        get_folds(self, horizon: int = 25):
            Создание разбиений данных для тестирования.
        get_model_result(self, y_pred_list: List[pd.Series]):
            Вычисление и вывод метрик для модели.
        get_visualisation(self):
            Визуализация фактических и прогнозируемых значений.
    '''
    def __init__(
            self, 
            data: Union[pd.Series, pd.DataFrame], 
            target_name: str = None,
            interpolate: bool = True,
            dropna: bool = True
            ) -> None:
        '''
        Инициализирует объект класса с данными временного ряда.

        Параметры:
            data (Union[pd.Series, pd.DataFrame]): Временной ряд в формате pd.Series или pd.DataFrame.
            target_name (str, опционально): Имя целевой переменной в случае pd.DataFrame.
            interpolate (bool): Необходимо ли интерполировать данные.
            dropna (bool): Необходимо ли удалить пропущенные значения.
        '''
        if not isinstance(data.index, pd.DatetimeIndex):
            print('Новый тип индексов: pd.DatetimeIndex')
            data.index = pd.to_datetime(data.index)
        
        if interpolate:
            data = data.interpolate(method='time')
        
        if dropna:
            data = data.dropna()

        data = data.sort_index()

        if isinstance(data, pd.Series):
            if target_name is not None:
                print("Предупреждение: target_name не нужно указывать для data: pd.Series")
            self.y = data
            self.X = pd.DataFrame(index=self.y.index)
        else:
            if target_name is None:
                raise ValueError("target_name обязательный параметр для data: pd.DataFrame")
            self.y = data[target_name]
            self.X = data.drop([target_name], axis=1)

        self.get_folds_was_called = False
        self.y_pred = None

        self.y_val = None
        self.X_val = None

    def add_lags(self, lags: List[int], drop_na: bool = True) -> None:
        '''
        Добавляет лаги (отставания) к временному ряду как новые признаки.

        Параметры:
            lags (List[int]): Список целых чисел, каждое из которых указывает на количество шагов отставания.
            drop_na (bool): Если True, то строки с пропущенными значениями после добавления лагов будут удалены.
        '''
        for lag in lags:
            self.X[f'lag_{lag}'] = self.y.shift(lag)

        if drop_na:
            self.X.dropna(inplace=True)
            self.y = self.y.loc[self.X.index]

        if self.get_folds_was_called:
            self.get_folds(horizon=self.horizon)

    def add_other_variables(self, other_variables: pd.DataFrame = None, date_variables: bool = False) -> None:
        '''
        Добавляет дополнительные внешние переменные к данным временного ряда.

        Параметры:
            other_variables (pd.DataFrame): DataFrame, содержащий внешние переменные для добавления.
            date_variables (bool): Будут ли добавлены переменные, связанные с датой.
        '''

        if other_variables is not None:
            if other_variables.isna().any().any():
                print('Предупреждение: в other_variables есть пропуски. Алгоритм не предусматривает их наличие')

            self.X = pd.concat([self.X, other_variables], axis=1, join='inner')


        if date_variables:
            self.X['days_since_start'] = (self.X.index - self.X.index[0]).days
            self.X['day'] = self.X.index.day
            self.X['dayofweek'] = self.X.index.dayofweek
            self.X['month'] = self.X.index.month
            self.X['year'] = self.X.index.year


        if self.get_folds_was_called:
            self.get_folds(horizon=self.horizon)

    def get_folds(
            self, 
            horizon: int = 25, 
            num_recent_folds: int = None, 
            validation_part: float = None
            ) -> None:
        '''
        Создает разбиения данных для кросс-валидации на основе указанного горизонта прогнозирования.

        Параметры:
            horizon (int): Горизонт прогнозирования, используемый для создания тестовых разбиений.
            num_recent_folds (int, optional): Количество последних разбиений для включения в результат.
                Если не указано, используются все разбиения.
            validation_part (float): Процентное отношение объема исходных данных, отводимое для валидационного набора. Этот параметр особенно полезен в случаях, когда оптимизация гиперпараметров модели требует значительного времени. Рекомендуется однократно настроить гиперпараметры на валидационном наборе и применять их для последующих разбиений (фолдов), избегая повторного обучения на каждом фолде.
        '''
        if validation_part is not None:
            if not 0 <= validation_part <= 1:
                raise ValueError("validation_part может быть только числом в отрезке от 0 до 1 включительно")
            
            last_val_obs = int(self.y.shape[0] * validation_part)

            self.y_val, self.y = self.y.iloc[:last_val_obs], self.y.iloc[last_val_obs:]
            self.X_val, self.X = self.X.iloc[:last_val_obs, :], self.X.iloc[last_val_obs:, :]

        self.horizon = horizon
        self.folds = []
        for i in range(0, len(self.X) - horizon, horizon):
            X_test = self.X.iloc[i:i + horizon]
            y_test = self.y.iloc[i:i + horizon]

            X_train = self.X.drop(X_test.index)
            y_train = self.y.drop(y_test.index)
            
            self.folds.append((X_train, y_train, X_test, y_test))
        
        if num_recent_folds is not None:
            self.folds = self.folds[-num_recent_folds:]
        
        self.get_folds_was_called = True

    def get_model_result(self, y_pred_list: List[pd.Series]) -> None:
        '''
        Вычисляет и выводит метрики качества модели на основе совокупности предсказаний.

        Параметры:
            y_pred_list (List[pd.Series]): Список pd.Series с прогнозами модели для каждого разбиения.
        '''
        self.y_pred = pd.concat(y_pred_list)
        self.y_true = self.y.loc[self.y_pred.index]

        self.mase = MASE(self.y_true, self.y_pred)
        self.smape = SMAPE(self.y_true, self.y_pred)
        self.mda = MDA(self.y_true, self.y_pred)
        self.dmda = DMDA(self.y_true, self.y_pred)
        self.zbmae = ZBMAE(self.y_true, self.y_pred)
        self.mae = mean_absolute_error(self.y_true, self.y_pred)

        print(f'MAE: {self.mae:.4f}')
        print(f'MASE: {self.mase:.4f}')
        print(f'ZBMAE: {self.zbmae:.4f}')
        print(f'SMAPE: {self.smape:.4f}%')
        print(f'MDA: {self.mda:.4f}%')
        print(f'DMDA: {self.dmda:.4f}%')

    def get_visualisation(self, start_date=None, end_date=None) -> None:
        '''
        Отображает визуализацию сравнения фактических значений временного ряда с прогнозными.
        
        Параметры:
            start_date (str, optional): Начальная дата среза для визуализации в формате 'YYYY-MM-DD'.
                                        Если None, визуализация начинается с первой доступной даты.
            end_date (str, optional): Конечная дата среза для визуализации в формате 'YYYY-MM-DD'.
                                      Если None, визуализация идет до последней доступной даты.
        '''
        if self.y_pred is None:
            raise ValueError('Метод get_visualisation должен вызываться после вызова метода get_model_result')
        
        y_true_sliced = self.y_true.copy()
        y_pred_sliced = self.y_pred.copy()
        
        if start_date:
            y_true_sliced = y_true_sliced[start_date:]
            y_pred_sliced = y_pred_sliced[start_date:]
        if end_date:
            y_true_sliced = y_true_sliced[:end_date]
            y_pred_sliced = y_pred_sliced[:end_date]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=y_true_sliced, label="Actual")
        sns.lineplot(data=y_pred_sliced, label="Predicted")
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.title("Actual vs Predicted")
        plt.legend()
        plt.show()


def train_and_evaluate(train_loader, X_test, model, optimizer, criterion, n_epochs=5):
    
    for epoch in range(n_epochs):
        model.train()
        for x_train, y_train in train_loader:
            y_pred = model(x_train)
            loss = criterion(y_train.view(-1, 1), y_pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    model.eval()

    y_pred = model(X_test).squeeze().detach().numpy()

    return y_pred

class Dataset(torch.utils.data.Dataset):
    """
    Our random dataset
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx, :], dtype=torch.float), self.y[idx]

class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        
        self.fc_1 = nn.Linear(15, 36) 
        self.batch_norm1 = nn.BatchNorm1d(8)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc_2 = nn.Linear(36, 8)
        self.batch_norm2 = nn.BatchNorm1d(8)
        # self.dropout2 = nn.Dropout(0.1)
        
        self.fc_3 = nn.Linear(8, 16)
        self.fc_4 = nn.Linear(16, 4)
        self.fc_5 = nn.Linear(4, 8)
        self.fc_6 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc_1(x))
        x = F.leaky_relu(self.fc_2(x))
        x = F.leaky_relu(self.fc_3(x))
        x = F.leaky_relu(self.fc_4(x))
        x = F.leaky_relu(self.fc_5(x))
        x = self.fc_6(x)

        return x


# ========================================================================== #

data = pd.read_csv('data/data_indexes_moex_log_returns.csv', index_col=0)

sampler = optuna.samplers.MOTPESampler()

smape_scorer = make_scorer(SMAPE)

for i in data.columns:
    ts = TimeSeriesProcessor(data[i])
    ts.add_lags(lags=[21, 22, 23, 24, 25, 26, 27, 30, 40, 50])
    ts.add_other_variables(date_variables=True)
    ts.get_folds(horizon=20, validation_part=0.3, num_recent_folds=20)


    # linear regression
    lr_model = LinearRegression()
    y_pred_list_lr = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        lr_model.fit(X_train, y_train)
        y_pred_lr = pd.Series(lr_model.predict(X_test), index=X_test.index)
        y_pred_list_lr.append(y_pred_lr)
    ts.get_model_result(y_pred_list_lr) 
    lr_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]


    # random forest
    X_val, y_val = ts.X_val, ts.y_val
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_int('max_features', 1, 10)
        }

        regressor = RandomForestRegressor(**param)
        return np.mean(
            cross_val_score(
                regressor, X_val, y_val, 
                n_jobs=-1, scoring=smape_scorer, 
                cv=5))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=30)
    rf_model = RandomForestRegressor(**study.best_params)
    y_pred_list_rf = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        rf_model.fit(X_train, y_train)
        y_pred_rf = pd.Series(rf_model.predict(X_test), index=X_test.index)
        y_pred_list_rf.append(y_pred_rf) 
    ts.get_model_result(y_pred_list_rf) 
    rf_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]


    # GB sklearn
    gb_model = GradientBoostingRegressor()
    y_pred_list_gb = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        gb_model.fit(X_train, y_train)
        y_pred_gb = pd.Series(gb_model.predict(X_test), index=X_test.index)
        y_pred_list_gb.append(y_pred_gb)
    ts.get_model_result(y_pred_list_gb) 
    gb_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]    
    
    # AdaBoost sklearn
    adaboost_model = AdaBoostRegressor()
    y_pred_list_adaboost = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        adaboost_model.fit(X_train, y_train)
        y_pred_adaboost = pd.Series(adaboost_model.predict(X_test), index=X_test.index)
        y_pred_list_adaboost.append(y_pred_adaboost)
    ts.get_model_result(y_pred_list_adaboost) 
    ada_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]  


    # Ridge
    ridge_model = Ridge()
    y_pred_list_ridge = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = pd.Series(ridge_model.predict(X_test), index=X_test.index)
        y_pred_list_ridge.append(y_pred_ridge)
    ts.get_model_result(y_pred_list_ridge)
    ridge_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]  


    # KNN
    knn_model = KNeighborsRegressor()
    y_pred_list_knn = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        knn_model.fit(X_train, y_train)
        y_pred_knn = pd.Series(knn_model.predict(X_test), index=X_test.index)
        y_pred_list_knn.append(y_pred_knn)
    ts.get_model_result(y_pred_list_knn)
    knn_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]  


    # XGBoost
    xgb_model = xgb.XGBRegressor()
    y_pred_list_xgb = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = pd.Series(xgb_model.predict(X_test), index=X_test.index)
        y_pred_list_xgb.append(y_pred_xgb)
    ts.get_model_result(y_pred_list_xgb)
    xgb_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]  

    # CatBoost
    def objective(trial):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 2, 8),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 20.0),
            'iterations': trial.suggest_int('iterations', 50, 800),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'loss_function': 'MAE',
            'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10),
            'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
            'od_type': 'Iter',
            'od_wait': trial.suggest_int('od_wait', 10, 70)
        }
        regressor = CatBoostRegressor(**param)
        return np.mean(
            cross_val_score(
                regressor, X_val, y_val, 
                n_jobs=-1, scoring=smape_scorer, 
                cv=5))
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=20)
    regressor = CatBoostRegressor(**study.best_params)
    y_pred_list_catboost = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds):
        regressor.fit(X_train, y_train, verbose=0)
        y_pred = pd.Series(
            regressor.predict(X_test),
            index=X_test.index
        )
        y_pred_list_catboost.append(y_pred) 
    ts.get_model_result(y_pred_list_catboost)
    catboost_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda]  


    # MPL
    improved_model = ImprovedModel()
    optimizer = torch.optim.Adam(improved_model.parameters())
    criterion = nn.HuberLoss()
    y_pred_list_mpl = []
    for X_train, y_train, X_test, y_test in tqdm(ts.folds): 
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        train = Dataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        y_pred = train_and_evaluate(train_loader, X_test_tensor, improved_model, optimizer, criterion, n_epochs=6)
        y_pred = pd.Series(
            y_pred,
            index=X_test.index
        )
        y_pred_list_mpl.append(y_pred) 
    ts.get_model_result(y_pred_list_mpl) 
    mpl_pred = [ts.mae, ts.mase, ts.zbmae, ts.smape, ts.mda, ts.dmda] 


    result = pd.DataFrame(
        {
            'lr': lr_pred,
            'knn': knn_pred,
            'ridge': ridge_pred,
            'rf': rf_pred,
            'gb': gb_pred,
            'adaboost': ada_pred,
            'xgb': xgb_pred,
            'catboost': catboost_pred,
            'mpl': mpl_pred
            
        },
        index=['mae', 'mase', 'zbmae', 'smape', 'mda', 'dmda']
    )

    result.to_csv(f'{i}_metrics.csv')
