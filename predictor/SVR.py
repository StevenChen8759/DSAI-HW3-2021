# LSTM Model
# Python Interpreter Module ImportError
import random
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np


# User Module Import

# Declare lambda function for calculating time
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
fetch_predict_date = lambda x : datetime_parse(x["time"][0]) + timedelta(days = 7)

# Function definition
def train(train_np_pair, test_np_pair, iplen=168, oplen=24):

    model_train = make_pipeline(StandardScaler(), SVR(epsilon=0.2, kernel='linear', coef0=0.0))

    X_train, Y_train = train_np_pair[0].reshape(len(train_np_pair[0]), 168), train_np_pair[1].reshape(len(train_np_pair[1]), 24)
    X_test, Y_test = test_np_pair[0].reshape(len(test_np_pair[0]), 168), test_np_pair[1].reshape(len(test_np_pair[1]), 24)


    clf = MultiOutputRegressor(model_train, n_jobs=7).fit(X_train, Y_train)

    infres = clf.predict(X_test)

    '''for i in range(len(infres)):
       infres[i] = infres[i] + np.random.uniform(low=0.0, high=50.0, size=(7,))'''

    # clf_all = MultiOutputRegressor(model_train).fit(X_train + X_test, Y_train + Y_test)

    logger.debug(f"Model: SVR, Testing Set RMSE -> {mean_squared_error(Y_test, infres, squared = False)}")

    # infres = clf_all.predict(X_test)
    # logger.debug(f"Model: SVR, Testing Set RMSE -> {mean_squared_error(Y_test, infres, squared = False)}")

    return clf


def load_model(filename):
    return joblib.load(filename)

def store_model(model, filename):
    joblib.dump(model, filename)

def inference(model, ip_data_df, iplen=168, oplen=24):
    # Consumption or generation data turns to Numpy ndarray
    ip_data_np = ip_data_df.to_numpy().reshape(1, 168)


    # Do inference
    result = model.predict(ip_data_np)

    # Return denormalized result
    # return denormalize_in(result)
    return result

def generate_bidding_one_day(input_df_con, input_df_gen):

    bidding_list = []

    # Phase 1 - Do inference of consumption and generation
    con_model = load_model("con_svr_model_v3.joblib")
    gen_model = load_model("gen_svr_model_v3.joblib")

    inf_con_data = inference(con_model, input_df_con["consumption"]).reshape(24, 1)
    inf_gen_data = inference(gen_model, input_df_gen["generation"]).reshape(24, 1)

    logger.debug(f"Inference consumption data shape: {inf_con_data.shape}")
    logger.debug(f"Inference generation data shape: {inf_gen_data.shape}")

    # Phase 2 - Get bidding date
    target_date = fetch_predict_date(input_df_con)

    for i in range(24):
        bid_timestamp = target_date + timedelta(hours=i)

        # Step 1: Fetch Hourly Predicted Consumption and Generation
        con_hr = round(inf_con_data[i][0], 2)
        gen_hr = round(inf_gen_data[i][0], 2)
        logger.debug(f"[{bid_timestamp:%Y-%m-%d %H:%M:%S}] -> Consumption: {con_hr:.2f} kWh, Generation: {gen_hr:.2f} kWh")

        # Step 2: Compare consumption and generation value
        if gen_hr <= 0.03 or bid_timestamp.hour < 6 or bid_timestamp.hour > 19:
            # Case 1: generation is almost zero, intentionally sell with higher price than city power
            logger.debug(f"Generate [sell] bidding - intentional case")
            templist = []
            templist.append(bid_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            templist.append("sell")
            templist.append(round(random.uniform(2.53, 2.75), 2)) # Randomly sell price
            templist.append(10.0)                                 # Sell 10.0 kWh
            bidding_list.append(templist)
        elif con_hr >= gen_hr:
            # Case 2: Consumption is bigger than generation, buy difference in price cheaper than city power
            logger.debug(f"Generate [buy] bidding - for cheaper price than city power")
            templist = []
            templist.append(bid_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            templist.append("buy")
            templist.append(round(random.uniform(2.38, 2.50), 2))   # Randomly buy price to make deal AMAP
            templist.append(round((con_hr - gen_hr), 2))            # Buy consumption minus generation kWh
            bidding_list.append(templist)
        elif con_hr < gen_hr:
            # Case 3: Generation is bigger than consumption, sell difference in price
            logger.debug(f"Generate [sell] bidding - good generate case")
            templist = []
            templist.append(bid_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            templist.append("sell")
            templist.append(round(random.uniform(2.32, 2.52), 2))   # Randomly sell price to make deal AMAP
            templist.append(round((gen_hr - con_hr), 2))            # Sell generation minus consumption kWh
            bidding_list.append(templist)

    return bidding_list
