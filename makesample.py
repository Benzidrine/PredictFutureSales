
from keras import layers
from pandas.core.frame import DataFrame
from tensorflow.python.keras.utils.data_utils import Sequence
import pandas as pd
import math
import keras
from generator import custom_generator
import matplotlib.pyplot as plt
import numpy as np
from save_load_model import load_model
import math

model = load_model("savedModel")

df : DataFrame = pd.read_csv("Data\\test_processed.csv")

dataset = custom_generator(df,100)
accuracy = 0
id_list = []
test_list = []
predict_list = []
i = 0

for batch in dataset:
    batchX = batch[0]['basic_input']
    ynew = model.predict(x=batchX,batch_size=100)
    if i % 1000:
        print(i,'done')
    for yhat, item in zip(ynew,batchX):
        id_list.append(i)
        test_list.append(item[1])
        predict_list.append(yhat[0])
        i += 1

df_new : DataFrame =  pd.DataFrame(
{
    'id': id_list,
    'test_list': test_list,
    'predict_list': predict_list
})

df_new.to_csv("Data\\sample_submission_test.csv", index=None)  

df_new : DataFrame =  pd.DataFrame(
{
    'ID': id_list,
    'item_cnt_month': predict_list
})

df_new.to_csv("Data\\sample_submission.csv", index=None)  