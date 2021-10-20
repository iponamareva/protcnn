Example of usage on EBI cluster: 

python3 run.py -na "exp-name" -tp 1.0 -al 1.0 -ml 1000 -fs 1100 -ks 9 -ep 100 -bs 32

main.py - main script. Loads data, creates and trains the model
layers.py - layers and ProtCNNModel
model utils.py - loss and metric
preprocessing.py - functions for loading data and creating datasets
constants.py - constants used throughout the script


