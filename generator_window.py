
from pandas.core.frame import DataFrame
from tensorflow.python.keras.utils.data_utils import Sequence
import pandas as pd
import numpy as np
import math

class custom_generator(Sequence):
    def __init__(self, dataframe : DataFrame, batch_size : int, window : int):
        """
        Initialise Dataset
        """
        super().__init__()
        self.shuffle = True
        self.dataframe = dataframe
        self.list_item_IDs = self.dataframe["item_id"].tolist()
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.list_IDs))
        # Generate Dictionaries
        self.shop_id_dict = {}
        self.shop_id_list = self.dataframe["shop_id"].tolist()
        self.item_id_dict = {}
        self.item_id_list = self.dataframe["item_id"].tolist()
        self.item_cnt_month_dict = {}
        self.item_cnt_month_list = self.dataframe["item_cnt_month"].tolist()
        self.cat_id_dict = {}
        self.cat_id_list = self.dataframe["cat_id"].tolist()
        self.data_block_num_dict = {}
        self.data_block_num_list = self.dataframe["data_block_num"].tolist()
        for id, shop_id, item_id, item_cnt, cat_id, data_block_num in zip(self.list_IDs, self.dataframe["shop_id"].tolist(), self.dataframe["item_id"].tolist(), self.dataframe["item_cnt_month"].tolist(), self.dataframe["cat_id"].tolist(), self.dataframe["data_block_num"].tolist()):
            self.shop_id_dict[id] = shop_id
            self.item_id_dict[id] = item_id
            self.item_cnt_month_dict[id] = item_cnt
            self.cat_id_dict[id] = cat_id
            self.data_block_num_dict[id] = data_block_num
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(math.floor((len(self.list_IDs) / self.batch_size)))
    
    def __getitem__(self, index):
        """
        Gets the indexes of batch_size number of data from list_IDs for one epoch
        If batch_size = 8, 8 files/indexes from list_ID are selected
        Makes sure that on next iteration, the batch does not come from same indexes as the previous batch
        The same batch is not seen again until __len()__ - 1 batches are done
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs)
        return X, y

    def __data_generation(self, list_IDs):
        # Creates an empty placeholder array that will be populated with data that is to be supplied
        X = np.empty((self.batch_size, 4))
        y = np.empty((self.batch_size, 1))
        
        for i, ID in enumerate(list_IDs):
            X[i,0] = self.shop_id_dict[ID]
            X[i,1] = self.item_id_dict[ID]
            X[i,2] = self.cat_id_dict[ID]
            X[i,3] = self.data_block_num_dict[ID]
            y[i,] = self.item_cnt_month_dict[ID]
        
        return { "basic_input": X }, y