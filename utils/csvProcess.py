# CSV data processor
# Python Interpreter Module ImportError
from datetime import datetime, timedelta

# Python External Module ImportError
import numpy as np
from loguru import logger

# User Module Import

# User defined global variable / lambda function in this Module

# Function Definition

def encode_tsdata(input_df, iplen=1, oplen=1):
    assert(iplen >= 1 and oplen >= 1)

    iplist_con = []
    iplist_gen = []
    oplist_con = []
    oplist_gen = []
    col_list = ["consumption", "generation"]
    for iter in range(iplen, len(input_df.index) - oplen + 1):
        ip_con_part = input_df["consumption"].loc[iter - iplen:iter - 1].to_numpy()
        ip_gen_part = input_df["generation"].loc[iter - iplen:iter - 1].to_numpy()
        op_con_part = input_df["consumption"].loc[iter:iter + oplen - 1].to_numpy()
        op_gen_part = input_df["generation"].loc[iter:iter + oplen - 1].to_numpy()

        iplist_con.append(ip_con_part)
        iplist_gen.append(ip_gen_part)
        oplist_con.append(op_con_part)
        oplist_gen.append(op_gen_part)

    # Input Data:  (N, 168, 1) * 2
    # Output Data: (N, 24, 1) * 2

    ret_ipdata_con = np.concatenate(iplist_con, axis=0).reshape(len(iplist_con), iplen, 1)
    ret_ipdata_gen = np.concatenate(iplist_gen, axis=0).reshape(len(iplist_gen), iplen, 1)
    ret_opdata_con = np.concatenate(oplist_con, axis=0).reshape(len(oplist_con), oplen, 1)
    ret_opdata_gen = np.concatenate(oplist_gen, axis=0).reshape(len(oplist_gen), oplen, 1)

    return ret_ipdata_con, ret_ipdata_gen, ret_opdata_con, ret_opdata_gen


def train_validation_split(input_nptuple, validation_ratio):

    data_in, data_out = input_nptuple

    trainingSize = round(len(data_in) * (1 - validation_ratio))
    validationSize = round(len(data_in)  * validation_ratio)

    logger.debug(f"Training Size: {trainingSize}, Validation Size: {validationSize}")

    train_in = data_in[validationSize:]
    train_out = data_out[validationSize:]

    validation_in = data_in[:validationSize]
    validation_out = data_out[:validationSize]

    return (train_in, train_out), (validation_in, validation_out)
