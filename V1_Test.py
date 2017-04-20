import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.preprocessing.image import flip_axis
from keras.layers import Input, Lambda, Convolution2D, Dense, Dropout
from keras.models import Model, model_from_json
from scipy import misc
import random
import json

