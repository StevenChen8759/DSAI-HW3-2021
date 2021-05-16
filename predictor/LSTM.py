# LSTM Model
# Python Interpreter Module ImportError
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense
from tensorflow.keras.callbacks import EarlyStopping

# User Module Import


# Function definition
# 336 to 48 LSTM Model, half(168->24) for predicting consumption, the other for predicting generation
def train(data_train_pair, data_validation_pair, iplen=168, oplen=24):
    # Note: Please input numpy ndarray as training data...
    trainingSize = len(data_train_pair[0])
    validationSize = len(data_validation_pair[0])
    logger.debug(f"Training Size: {trainingSize}, Validation Size: {validationSize}")

    train_in, train_out = data_train_pair

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(iplen, 1), return_sequences=False))
    # output shape: (24, 2)
    # model.add(TimeDistributed(Dense(4)))
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(Dense(1))
    model.add(RepeatVector(oplen))
    model.compile(loss="mse", optimizer="adam")

    logger.debug("Model structure...")
    model.summary()

    logger.debug("Fitting Model, please wait for a moment...")
    callback = EarlyStopping(monitor="val_loss", patience=1, verbose=1, mode="auto")
    fitinfo = model.fit(train_in, train_out, epochs=50, batch_size=1024, verbose=1, validation_data=data_validation_pair, callbacks=[callback])
    cntEpoch = len(fitinfo.history['val_loss'])
    logger.debug(f"Final Epoch Count: {cntEpoch}, Training Loss: {fitinfo.history['loss'][cntEpoch - 1]:.4f}, Validation Loss: {fitinfo.history['val_loss'][cntEpoch - 1]:.4f}")
    return model

def load_model(filename):
    return tensorflow.keras.models.load_model(filename)

def store_model(model, filename):
    model.save(filename)

def inference(model, ip_con_gen, iplen=168, oplen=24):
    # Consumption and generation data turns to Numpy ndarray
    con_gen_inf = ip_con_gen.to_numpy().reshape(len(ip_con_gen.index), 168, 2)

    # Do inference
    result = model.predict(con_gen_inf, batch_size=8, verbose=1)

    return result
