import os
import tensorflow as tf

def save_model(model, name_of_model):
    model.save(os.path.join("savedmodels",name_of_model))
    print("saved model")

def load_model(name_of_model):
    model = tf.keras.models.load_model(os.path.join("savedmodels",name_of_model))
    print("model loaded")
    return model