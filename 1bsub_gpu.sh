#!/bin/bash

bsub -M 30000 -P gpu -gpu - "python3 main.py"


