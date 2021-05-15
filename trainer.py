# Trainer for specific model
# Python Interpreter Module ImportError
import argparse
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger

# User Module Import
from utils import csvIO, visualizer
from predictor import random_model, fbrb_model

# User defined global variable / lambda function in this Module
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# Function Definition
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="select this argument to train model")
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

# Main function
if __name__ == "__main__":
    args = config()