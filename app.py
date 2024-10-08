
##
from datawrangler import DataWrangler 
from modeltrainer import ModelTrainer
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

import streamlit as st
import pandas as pd
import numpy as np
import joblib


spark = SparkSession.builder.appName("DataWranglingApp").getOrCreate()
data_wrangler = DataWrangler(spark)

st.title("Predict the number of products sold")

shop_id = st.number_input("Enter Shop ID", min_value=0, step=1)
item_id = st.number_input("Enter Item ID", min_value=0, step=1)


if st.button("Predict Sales"):
    if shop_id is not None and item_id is not None:
        input_data = pd.DataFrame({
            'shop_id': [shop_id],
            'item_id': [item_id],
            'date_block_num': [34],
            'item_cnt_day': [0]
        })
        test = spark.createDataFrame(input_data)
        processed_train = spark.read.csv('./processed_train.csv', header=True, inferSchema=True)

        try:
            model = joblib.load('./trained_model.pkl')  
        except FileNotFoundError:
            st.error('Can not find model')
            st.stop()


        processed_test = data_wrangler.process_test_data(processed_train, test)
        processed_test = processed_test.filter(processed_test['date_block_num'] == 34) 

        processed_test_data = processed_test.drop('item_cnt_month').toPandas()

        predictions = model.predict(processed_test_data)
        predicted_sales = predictions['prediction'].item() 
        
        st.success(f"Predicted total sales: {predicted_sales} units")

    else:
        st.warning("Please enter Shop ID and Item ID.")

