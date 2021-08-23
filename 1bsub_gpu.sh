#!/bin/bash

bsub -M 20000 -P gpu -gpu - 'python3 main.py -na "test_exp_3" -ep 1'


