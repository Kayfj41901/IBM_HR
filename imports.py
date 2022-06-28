import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import MinMaxScaler 
# Import for modeling
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import prepare

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import visuals
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats
import imports

pd.options.display.max_rows = 100 

# default pandas decimal number display format
pd.options.display.float_format = '{:20,.2f}'.format
