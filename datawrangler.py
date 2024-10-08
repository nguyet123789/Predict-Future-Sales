##
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as F_sum, min as F_min, max as F_max, last as F_last, stddev as F_stddev
from pyspark.sql import Window
from itertools import product
from pyspark.sql import functions as F

class DataWrangler:
    def __init__(self, spark):
        self.spark = spark

    def _data_cleaning(self, df, flag=False):
        df = df.select('shop_id', 'item_id', 'date_block_num', 'item_cnt_day')
            
        df = df.withColumn('item_cnt_day', when(col('item_cnt_day') < 0, 0).otherwise(col('item_cnt_day')))
        return df

    def _sample_shops_items(self, df, shops, items, num_shops=60, num_items=25000):
        total_shops = shops.count()
        total_items = items.count()

        sampled_shop_ids = [row.shop_id for row in shops.select('shop_id').distinct()
                                .sample(False, num_shops / total_shops).collect()]

        sampled_item_ids = [row.item_id for row in df.select('item_id').distinct()
                                .sample(False, num_items / total_items).collect()]

        return sampled_shop_ids, sampled_item_ids

    def _create_combinations(self, df, sampled_shop_ids=None, sampled_item_ids=None):
        if sampled_shop_ids is None:
            sampled_shop_ids = [row.shop_id for row in df.select('shop_id').distinct().collect()]
        if sampled_item_ids is None:
            sampled_item_ids = [row.item_id for row in df.select('item_id').distinct().collect()]

        shop_item_combinations = self.spark.createDataFrame(
            list(product(sampled_shop_ids, sampled_item_ids)),
            schema=['shop_id', 'item_id']
        )
        date_block_nums = df.select('date_block_num').distinct()
        combinations = date_block_nums.crossJoin(shop_item_combinations)
        data_combinations = combinations.select('date_block_num', 'shop_id', 'item_id')
        return data_combinations

    def _calculate_monthly_sales(self, df, data_combinations=None):
        monthly_sales = (
            df.groupBy(['shop_id', 'item_id', 'date_block_num'])
            .agg(F_sum('item_cnt_day').alias('item_cnt_month'))
        )
        if data_combinations is not None:
            data_with_sales = data_combinations.join(
                monthly_sales,
                on=['shop_id', 'item_id', 'date_block_num'],
                how='left'
            )
        else: 
            data_with_sales = monthly_sales
        data_with_sales = data_with_sales.fillna({'item_cnt_month': 0})
        data_with_sales = data_with_sales.select('shop_id', 'item_id', 'date_block_num', 'item_cnt_month')
        return data_with_sales
    

    def _generate_aggregations_over_time_frames(self, agg_func, input_column, output, order_by_cols):
        aggregation_info = [
            ((0, 0), f'{output}_last_1m'),  # Last 1 month
            ((-1, 0), f'{output}_last_2m'),  # Last 2 months
            ((-2, 0), f'{output}_last_3m'),  # Last 3 months
            ((-3, 0), f'{output}_last_4m'),  # Last 4 months
            ((-4, 0), f'{output}_last_5m'),  # Last 5 months
        ]
    
        window = Window.partitionBy('shop_id', 'item_id').orderBy(order_by_cols)
    
        aggregations = []
        for agg in aggregation_info:
            aggregations.append(
                agg_func(F.col(input_column))
                .over(window.rowsBetween(agg[0][0], agg[0][1]))
                .alias(agg[1])
            )

        return aggregations

    def _generate_features_for_input_column(self, input_column, order_by_cols):
        return [
            *self._generate_aggregations_over_time_frames(F.min, input_column, f'min_{input_column}', order_by_cols),
            *self._generate_aggregations_over_time_frames(F.max, input_column, f'max_{input_column}', order_by_cols),
            *self._generate_aggregations_over_time_frames(F.last, input_column, f'last_{input_column}', order_by_cols),
            *self._generate_aggregations_over_time_frames(F.stddev, input_column, f'std_{input_column}', order_by_cols),
        ]
        
    def _generate_target(self, input_column, output, order_by_cols):
        aggregation_info = [
            ((1, 1), f'{output}_1m'),
        ]
    
        window = Window.partitionBy('shop_id', 'item_id').orderBy(order_by_cols)
    
        aggregations = []
        for agg in aggregation_info:
            aggregations.append(
                F.sum(F.col(input_column))
                .over(window.rowsBetween(agg[0][0], agg[0][1]))
                .alias(agg[1])
        )
    
        return aggregations

    def _extract_historical_data(self, train_data, test_data):
        min_test_date_block_num = test_data.agg({"date_block_num": "min"}).collect()[0][0]
        
        historical_data = train_data.filter(
            (col('date_block_num') >= (min_test_date_block_num - 5)) & 
            (col('date_block_num') < min_test_date_block_num)
        )
        
        test_shop_item = test_data.select('shop_id', 'item_id').distinct()
        
        historical_data = historical_data.join(
            test_shop_item,
            on=['shop_id', 'item_id'],
            how='inner'
        )
        
        return historical_data

    def _process_data(self, data_to_train):
        ORDER_BY_COLS = ['shop_id', 'item_id', 'date_block_num']

        processed_data = data_to_train.select("*",
                                              *self._generate_features_for_input_column('item_cnt_month', ORDER_BY_COLS),
                                              *self._generate_target('item_cnt_month', 'target', ORDER_BY_COLS))
        return processed_data

    def prepare_data(self, processed_train, shops, items, num_shops=60, num_items=1000):
        X = self._data_cleaning(processed_train)

        sampled_shop_ids, sampled_item_ids = self._sample_shops_items(X, shops, items, num_shops, num_items)

        combination = self._create_combinations(X, sampled_shop_ids, sampled_item_ids)

        processed_data = self._calculate_monthly_sales(X, combination)

        data_to_train = self._process_data(processed_data)

        return data_to_train

    def process_test_data(self, train_data, test_data):
        processed_train = self._data_cleaning(train_data, True)
        
        historical_data = self._extract_historical_data(processed_train, test_data)
        
        combined_data = historical_data.unionByName(test_data)

        processed_test_data = self._calculate_monthly_sales(combined_data)

        data_to_test = self._process_data(processed_test_data)

        return data_to_test

