import os
import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description="Arguments for run")

parser.add_argument("-M", "--memory", nargs="?", type=int, default=20000)
parser.add_argument("-na", "--exp-name", nargs="?", type=str, default="exp")
parser.add_argument("-ep", "--epochs", nargs="?", type=int, default=100)
parser.add_argument("-tp", "--true-prop", nargs="?", type=float, default=0.0)
parser.add_argument("-lr", "--learning-rate", nargs="?", type=float, default=0.0003)
parser.add_argument("-fs", "--num-filters", nargs="?", type=int, default=384)
parser.add_argument("-al", "--alpha", nargs="?", type=float, default=1.0)

args = parser.parse_args()
dateTimeObj = datetime.now()
exp_name = args.exp_name
exp_name += "_"
exp_name += dateTimeObj.strftime("%d-%b-%Y_%H-%M-%S")



command = f'bsub -M {args.memory} -P gpu -gpu - "python3 main.py -na {exp_name} -ep {args.epochs} -tp {args.true_prop} -lr {args.learning_rate} -fs {args.num_filters} -al {args.alpha}"'
print("***Executing:", command)
os.system(command)
