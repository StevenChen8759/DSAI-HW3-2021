# LSTM Model
# Python Interpreter Module ImportError
import random
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense, Conv1D, BatchNormalization, Dropout, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping

# User Module Import

# Declare lambda function for calculating time
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
fetch_predict_date = lambda x : datetime_parse(x["time"][0]) + timedelta(days = 7)

def normalize_in(input_np, input_range=(0,12)):
    ip_min, ip_max = input_range
    norm_mapper = np.vectorize(lambda x: (x - ip_min) / (ip_max - ip_min))
    return norm_mapper(input_np)

def denormalize_in(input_np, original_input_range=(0,12)):
    ip_min, ip_max = original_input_range
    denorm_mapper = np.vectorize(lambda x: x * (ip_max - ip_min) + ip_min)
    return denorm_mapper(input_np)

def consumption_model_struct(iplen, oplen):
    model = Sequential()
    model.add(LSTM(50, activation='relu',input_shape=(iplen, 1), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(RepeatVector(oplen))                               # output shape: (24, 1)
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='linear')))    # output shape: (24, 1)
    return model

def generation_model_struct(iplen, oplen):
    model = Sequential()
    model.add(LSTM(50, activation='relu',input_shape=(iplen, 1), return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(RepeatVector(oplen))                               # output shape: (24, 1)
    model.add(LSTM(50, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation='linear')))    # output shape: (24, 1)
    return model

# Function definition
'''def train(data_train_pair, data_validation_pair, struct_type,  iplen=168, oplen=24):
    # Note: Please input numpy ndarray as training data...
    trainingSize = len(data_train_pair[0])
    validationSize = len(data_validation_pair[0])
    logger.debug(f"Training Size: {trainingSize}, Validation Size: {validationSize}")

    train_in, train_out = data_train_pair

    if struct_type == "consumption":
        model = consumption_model_struct(iplen, oplen)
    elif struct_type == "generation":
        model = generation_model_struct(iplen, oplen)
    model.compile(loss="mse", optimizer="adam")

    logger.debug("Model structure...")
    model.summary()

    logger.debug("Fitting Model, please wait for a moment...")
    callback = EarlyStopping(monitor="val_loss", patience=1, verbose=1, mode="auto")
    fitinfo = model.fit(train_in, train_out, epochs=200, batch_size=1024, verbose=1, validation_data=data_validation_pair, callbacks=[callback])
    cntEpoch = len(fitinfo.history['val_loss'])
    logger.debug(f"Final Epoch Count: {cntEpoch}, Training Loss: {fitinfo.history['loss'][cntEpoch - 1]:.4f}, Validation Loss: {fitinfo.history['val_loss'][cntEpoch - 1]:.4f}")
    return model'''

# Function definition
def train_2(data_pair, struct_type, iplen=168, oplen=24):
    # Note: Please input numpy ndarray as training data...
    inputSize = len(data_pair[0])
    logger.debug(f"Training Size: {inputSize}, Validation Ratio: 0.25")

    # train_in, train_out = normalize_in(data_pair[0]), normalize_in(data_pair[1])
    train_in, train_out = data_pair

    if struct_type == "consumption":
        model = consumption_model_struct(iplen, oplen)
    elif struct_type == "generation":
        model = generation_model_struct(iplen, oplen)
    model.compile(loss="mse", optimizer="adam")

    logger.debug(f"Model structure for {struct_type}...")
    model.summary()

    logger.debug(f"Fitting {struct_type} Model, please wait for a moment...")
    callback = EarlyStopping(monitor="val_loss", patience=20, verbose=1, mode="auto")
    fitinfo = model.fit(train_in, train_out, epochs=200, batch_size=8192, verbose=1, validation_split=0.25, shuffle=True, callbacks=[callback])
    cntEpoch = len(fitinfo.history['loss'])
    logger.debug(f"Final Epoch Count: {cntEpoch}, Training Loss: {fitinfo.history['loss'][cntEpoch - 1]:.4f}, Validation Loss: {fitinfo.history['val_loss'][cntEpoch - 1]:.4f}")
    return model

def load_model(filename):
    return tf.keras.models.load_model(filename)

def store_model(model, filename):
    model.save(filename)

def inference(model, ip_data_df, iplen=168, oplen=24):
    # Consumption or generation data turns to Numpy ndarray
    ip_data_np = ip_data_df.to_numpy().reshape(1, 168, 1)


    # Do inference
    result = model.predict(ip_data_np, batch_size=64, verbose=1)

    # Return denormalized result
    # return denormalize_in(result)
    return result

def generate_bidding_one_day(input_df_con, input_df_gen):

    bidding_list = []

    # Phase 1 - Do inference of consumption and generation
    con_model = load_model("con_lstm_model_v2.h5")
    gen_model = load_model("gen_lstm_model_v2.h5")

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
        gen_hr = round(inf_con_data[i][0], 2)
        logger.debug(f"[{bid_timestamp:%Y-%m-%d %H:%M:%S}] -> Consumption: {con_hr:.2f} kWh, Generation: {gen_hr:.2f} kWh")

        # Step 2: Compare consumption and generation value
        if gen_hr <= 0.03:
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
            templist = []
            templist.append(bid_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
            templist.append("sell")
            templist.append(round(random.uniform(2.38, 2.50), 2))   # Randomly sell price to make deal AMAP
            templist.append(round((gen_hr - con_hr), 2))            # Sell generation minus consumption kWh
            bidding_list.append(templist)

    return bidding_list
