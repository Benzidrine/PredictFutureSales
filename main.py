
from keras import layers
from pandas.core.frame import DataFrame
from tensorflow.python.keras.utils.data_utils import Sequence
import pandas as pd
import math
import keras
from generator import custom_generator
import matplotlib.pyplot as plt
from save_load_model import save_model

df : DataFrame = pd.read_csv("Data\\sales_train_processed.csv")
df = df[df["data_block_num"] >= 4]

batch_size : int = 100
epochs : int = 1

data_length : int = len(df)
train_size : int = math.floor(data_length * 1)
# TEMP 
# train_size : int = math.floor(data_length * 1)
valid_size : int = math.floor(data_length * 0.2)

train_df = df[:train_size]
valid_df = df[train_size:(train_size + valid_size)]
test_df = df[(train_size + valid_size):]

# create data generators
image_datasets = {x[0]: custom_generator(x[1],batch_size)
    for x in [('train',train_df), ('val',valid_df), ('test',test_df)]}

fc_layer_size = 40
# simple dense model
inputs = keras.Input(shape=(8,1), name='basic_input')
outputs = layers.SimpleRNN(64, input_shape=(6,1), return_sequences=True)(inputs)
outputs = layers.SimpleRNN(32, return_sequences=True)(outputs) #recurrent layer 2, 32 neurons
outputs = layers.SimpleRNN(16)(outputs) #recurrent layer 3, 16 neurons
outputs = layers.Dense(8, activation='tanh', name='second_layer')(outputs)
#outputs = layers.SimpleRNN(10, input_shape=(6,1), return_sequences=True)(inputs)
outputs = layers.Dense(1, activation='linear', name='output_layer')(outputs)

model = keras.Model(inputs=[inputs], outputs=outputs)
print(model.summary())
opt = keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
#opt = keras.optimizers.Adadelta()
model.compile(optimizer=opt,  # Optimizer
            # Loss function to minimize
            loss="mean_squared_error",
            # List of metrics to monitor
            metrics=["mean_absolute_error"])


H : keras.callbacks.History = model.fit( #type: ignore
    x=image_datasets["train"],
    steps_per_epoch=(len(image_datasets["train"])),
    validation_data=image_datasets["val"],
    validation_steps=(len(image_datasets["val"])),
    epochs=epochs) 

def plot_history(H):
    loss = H.history['loss']
    val_loss = H.history['val_loss']
    epochs = range(1, len(loss)+1)
    plt.plot(epochs,loss,'bo',label='training loss')
    plt.plot(epochs,val_loss,'b',label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

save_model(model, "savedModel")

plot_history(H)