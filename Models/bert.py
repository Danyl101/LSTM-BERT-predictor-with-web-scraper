import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import numpy as np  

data=pd.read_csv("data.csv") #Load the data from csv file
