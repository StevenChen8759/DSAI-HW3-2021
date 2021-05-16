# Trainer for specific model
# Python Interpreter Module ImportError
import argparse
from datetime import datetime, timedelta

# Python External Module ImportError
import numpy as np
from loguru import logger

# User Module Import
from utils import csvIO, csvProcess, visualizer
from predictor import LSTM

# User defined global variable / lambda function in this Module
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# Function Definition
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_encoded", action="store_true", help="read pre-processed npy file")
    parser.add_argument("--ip_folder", default="./training_data", help="input training data of predicting consumption and generation")
    parser.add_argument("--op_model", default="congen_model.csv", help="output model name of predicting consumption and generation")

    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = config()

    # Read data and encode to time series format
    if args.read_encoded:
        household_con_np_in = np.load("household_in_con.npy")
        household_gen_np_in = np.load("household_in_gen.npy")
        household_con_np_out = np.load("household_out_con.npy")
        household_gen_np_out = np.load("household_out_gen.npy")
    else:
        household_con_input = []
        household_gen_input = []
        household_con_output = []
        household_gen_output = []
        for i in range(50):
            logger.info(f"Reading {args.ip_folder}/target{i}.csv")
            idv_house = csvIO.read(f"{args.ip_folder}/target{i}.csv")

            ipdata_con, ipdata_gen, opdata_con, opdata_gen = csvProcess.encode_tsdata(idv_house, 168, 24)
            household_con_input.append(ipdata_con)
            household_gen_input.append(ipdata_gen)
            household_con_output.append(opdata_con)
            household_gen_output.append(opdata_gen)

        household_np_data_len = len(household_con_input[0]) * len(household_con_input)
        household_con_np_in = np.concatenate(household_con_input, axis=0).reshape(household_np_data_len, 168, 1)
        household_gen_np_in = np.concatenate(household_gen_input, axis=0).reshape(household_np_data_len, 168, 1)
        household_con_np_out = np.concatenate(household_con_output, axis=0).reshape(household_np_data_len, 24, 1)
        household_gen_np_out = np.concatenate(household_gen_output, axis=0).reshape(household_np_data_len, 24, 1)

        np.save("household_in_con.npy", household_con_np_in)
        np.save("household_in_gen.npy", household_gen_np_in)
        np.save("household_out_con.npy", household_con_np_out)
        np.save("household_out_gen.npy", household_gen_np_out)

    logger.debug(f"Time Series Input data Shape (For Both): {household_con_np_in.shape}")
    logger.debug(f"Time Series Output data Shape (For Both): { household_con_np_out.shape}")



    # Split training set and testing set
    ts_train_con , ts_validation_con = csvProcess.train_validation_split((household_con_np_in, household_con_np_out), 0.20)
    final_consumption_model = LSTM.train(ts_train_con, ts_validation_con)
    LSTM.store_model(final_consumption_model, "con_lstm_model_v1.h5")

    ts_train_gen , ts_validation_gen = csvProcess.train_validation_split((household_gen_np_in, household_gen_np_out), 0.20)
    final_generation_model = LSTM.train(ts_train_gen, ts_validation_gen)
    LSTM.store_model(final_generation_model, "gen_lstm_model_v1.h5")

    if args.read_encoded:
        household_con_np_in.close()
        household_gen_np_in.close()
        household_con_np_out.close()
        household_gen_np_out.close()
