# ProtENN distillation

Example of usage on EBI cluster: 

```
python3 run.py -na "exp-name" -tp 1.0 -al 1.0 -ml 1000 -fs 1100 -ks 9 -ep 100 -bs 32
```
### Description of parameters

- tp - true proportion, proportion of the true labels in the loss function
- al - alpha, the power to which the teacher predictions are normalized. If power is x, alpla should be 1/x
- ml - maximal length of the sequence
- fs - number of filters
- ks - kernel size
- ep - number of epochs
- bs - batch size

### Description of scripts

- main.py - main script. Loads data, creates and trains the model
- layers.py - layers and ProtCNNModel
- model_utils.py - loss and metric
- preprocessing.py - functions for loading data and creating datasets
- constants.py - constants used throughout the script


