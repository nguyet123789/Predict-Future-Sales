##
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error


class ModelTrainer:
    def __init__(self):
        self.model = None  

    def create_backtest_splits(self, df, n_splits=5, validation_months=2, test_months=1, date_column='date_block_num'):
        df = df.sort_values(date_column)

        months = df[date_column].unique()
        total_months = len(months) - 1
        split_size = validation_months + test_months
        max_train_end = total_months - n_splits * split_size

        if max_train_end <= 0:
            raise ValueError("The number of months of data is not enough to create splits for the backtest.")

        splits = []
        for i in range(n_splits):
            train_end = max_train_end + i * split_size
            val_start = train_end
            val_end = val_start + validation_months
            test_start = val_end
            test_end = test_start + test_months

            train_months = months[:train_end]
            val_months = months[val_start:val_end]
            test_months_list = months[test_start:test_end]

            train_data = df[df[date_column].isin(train_months)]
            val_data = df[df[date_column].isin(val_months)]
            test_data = df[df[date_column].isin(test_months_list)]

            splits.append((train_data, val_data, test_data))

        return splits

    def smape(self, y_true, y_pred):
        return 100 * np.mean(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-8))

    def train_model(self, processed_data, n_splits=5, validation_months=2, test_months=1):
        if not isinstance(processed_data, pd.DataFrame):
            processed_data = processed_data.toPandas()

        splits = self.create_backtest_splits(processed_data, n_splits=n_splits, validation_months=validation_months, test_months=test_months, date_column='date_block_num')

        results = []
        self.model = LGBMRegressor(n_estimators=5000, learning_rate=0.01, max_depth=32, num_leaves=128)

        for i, (train_data, val_data, test_data) in enumerate(splits):
            print(f"\nBacktest lần {i + 1}")

            y_train = train_data['target_1m']
            X_train = train_data.drop(['date_block_num', 'target_1m'], axis=1)

            y_val = val_data['target_1m']
            X_val = val_data.drop(['date_block_num', 'target_1m'], axis=1)

            y_test = test_data['target_1m']
            X_test = test_data.drop(['date_block_num', 'target_1m'], axis=1)

            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            X_test = X_test.fillna(0)

            self.model.fit(
                X_train,
                y_train,
                eval_metric='rmse',
                eval_set=[(X_val, y_val)],
                callbacks=[early_stopping(stopping_rounds=500)],
            )

            best_iteration = self.model.best_iteration_
            print(f"Best number of loops: {best_iteration}")

            y_pred = np.round(self.model.predict(X_test))
            smape_value = self.smape(y_test.values, y_pred)
            rmse = root_mean_squared_error(y_test.values, y_pred)
            print(f"SMAPE on test set: {smape_value}%")
            print(f"RMSE on test set: {rmse}%")

            results.append({
                'backtest': i + 1,
                'best_iteration': best_iteration,
                'smape': smape_value,
                'rsme': rmse
            })

        return results

    def predict(self, processed_test_data):
        if not isinstance(processed_test_data, pd.DataFrame):
            test_df = processed_test_data.toPandas()
        else:
            test_df = processed_test_data.copy()

        X_test = test_df.drop(['date_block_num', 'target_1m', 'shop_id', 'item_id'], axis=1, errors='ignore')
        X_test = X_test.fillna(0)

        predictions = np.round(self.model.predict(X_test))

        test_df['prediction'] = predictions

        return test_df[['shop_id', 'item_id', 'date_block_num', 'prediction']]

