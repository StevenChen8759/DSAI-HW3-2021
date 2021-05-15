# Python Interpreter Module ImportError
import argparse
from datetime import datetime, timedelta

# Python External Module ImportError
from loguru import logger

# User Module Import
from utils import csvIO, visualizer
from model import random_model, fbrb_model

# User defined global variable / lambda function in this Module
datetime_parse = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")

# Function Definition
# You should not modify this part.
def config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()



if __name__ == "__main__":

    args = config()
    logger.debug(f"Consumption input csv file    -> ./datasets/{args.consumption}")
    logger.debug(f"Generation input csv file     -> ./datasets/{args.generation}")
    logger.debug(f"Bidding result input csv file -> ./datasets/{args.bidresult}")
    logger.debug(f"Model Prediction Output       -> ./{args.output}")

    consumption = csvIO.read(args.consumption)
    generation = csvIO.read(args.generation)
    bidresult = csvIO.read(args.bidresult)

    logger.info(f"consumption length: {len(consumption.index)}")
    logger.info(f"generation length: {len(generation.index)}")
    logger.info(f"bidresult is not null: {not bidresult.empty}")



    data = random_model.generate_bidding_one_day(consumption, generation)
    csvIO.write(args.output, data)
