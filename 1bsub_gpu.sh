#!/bin/bash

bsub -M 20000 -P gpu -gpu - "python3 main.py"


