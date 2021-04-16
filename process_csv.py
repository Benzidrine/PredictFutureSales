from numpy.lib.function_base import append
from numpy.lib.shape_base import split
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

def add_ID_field(csv_file_location : str, new_csv_file_name : str) -> None:
    """
    add ID field to new csv file
    """
    df : DataFrame = pd.read_csv(csv_file_location)
    ids_list : list = []
    i : int = 0
    for row in df.iterrows():
        # show some progress
        if (i % 10000 == 0 and i > 0):
            print(i,"done")
        ids_list.append(i)
        i += 1
    ids_series : Series = pd.Series(ids_list)
    df["ID"] = ids_series
    df.to_csv(new_csv_file_name)

def create_test_csv(csv_file_location : str, csv_item_categories_location : str, sales_train_location : str, new_csv_file_name : str) -> None:
    """
    Make a csv file that contains all needed data for the test set so that the sample submission can be made
    """
    df : DataFrame = pd.read_csv(csv_file_location)
    df_items : DataFrame = pd.read_csv(csv_item_categories_location)
    dict_new : dict = {}
    shop_id_list : list = df["shop_id"].tolist()
    item_id_list : list = df["item_id"].tolist()
    i : int = 0
    for shop_id, item_id in zip(shop_id_list, item_id_list):
        key = "34_" + str(shop_id) + "_" + str(item_id)
        dict_new[key] = 0
            
    # main csv where we can develop the lag1 feature
    df_main : DataFrame = pd.read_csv(sales_train_location)
    # create a dictionary for lag1
    df_main = df_main[df_main["data_block_num"] >= 30]
    lag_1_dict = {}
    item_id_list: list = df_main["item_id"].tolist()
    item_cnt_list : list = df_main["item_cnt_month"].tolist()    
    shop_id_list : list = df_main["shop_id"].tolist()    
    date_block_list : list = df_main["data_block_num"]
    for item_id, item_cnt, shop_id, date_block in zip(item_id_list, item_cnt_list, shop_id_list, date_block_list):
        #needs shop added
        lag_1_dict[str(date_block) + "_" + str(item_id) + "_" + str(shop_id)] = item_cnt
    # make category dictionary
    df_item_id_cat_list : list = df_items["item_id"].tolist()
    df_cat_id_cat_list : list = df_items["item_category_id"].tolist()
    cat_dict : dict = {}
    item_id_cat : int
    cat_id : int
    for item_id_cat, cat_id in zip(df_item_id_cat_list,df_cat_id_cat_list):
        cat_dict[item_id_cat] = cat_id
    
    # create new dataframe
    i : int = 0
    id_list : list = []
    data_block_num_list : list = []
    shop_id_list : list = []
    item_id_list : list = []
    cat_id_list : list = []
    item_cnt_month_list : list = []
    lag_one_list : list = []
    lag_two_list : list = []
    lag_three_list : list = []
    lag_four_list : list = []
    key : str
    value : float
    for key, value in dict_new.items():
        # show progress
        if (i % 10000 == 0 and i > 0):
            print(i,"writing done") 
        data_block_num : str
        shop_id : str
        item_id : str 
        data_block_num, shop_id, item_id = key.split("_")
        id_list.append(i)
        data_block_num_list.append(int(data_block_num))
        shop_id_list.append(int(shop_id))
        item_id_list.append(int(item_id))
        cat_id_list.append(int(cat_dict[int(item_id)]))
        item_cnt_month_list.append(float(0))
        lag_one_key = "33_" + str(item_id) + "_" + str(shop_id)
        lag_two_key = "32_" + str(item_id) + "_" + str(shop_id)
        lag_three_key = "31_" + str(item_id) + "_" + str(shop_id)
        lag_four_key = "30_" + str(item_id) + "_" + str(shop_id)
        if (lag_one_key in lag_1_dict.keys()):
            lag_one_list.append(lag_1_dict[lag_one_key])
        else:
            lag_one_list.append(0)
        if (lag_two_key in lag_1_dict.keys()):
            lag_two_list.append(lag_1_dict[lag_two_key])
        else:
            lag_two_list.append(0)
        if (lag_three_key in lag_1_dict.keys()):
            lag_three_list.append(lag_1_dict[lag_three_key])
        else:
            lag_three_list.append(0)
        if (lag_four_key in lag_1_dict.keys()):
            lag_four_list.append(lag_1_dict[lag_four_key])
        else:
            lag_four_list.append(0)
        i += 1

    df_new : DataFrame =  pd.DataFrame(
    {
        'id': id_list,
        'data_block_num': data_block_num_list,
        'shop_id': shop_id_list,
        'item_id': item_id_list,
        'cat_id': cat_id_list,
        'item_cnt_month': item_cnt_month_list,
        'lag_one': lag_one_list,
        'lag_two': lag_two_list,
        'lag_three': lag_three_list,
        'lag_four': lag_four_list
    })
    # save new dataframe
    df_new.to_csv(new_csv_file_name, index=None)  
    
def convert_to_month_format(csv_file_location : str, csv_item_categories_location : str, new_csv_file_name : str) -> None:
    """
    take the sales_train.csv and change it to a monthly format and process feature engineering
    """
    df : DataFrame = pd.read_csv(csv_file_location)
    df_items : DataFrame = pd.read_csv(csv_item_categories_location)
    dict_new : dict = {}
    data_block_num_list : list = df["date_block_num"].tolist()
    shop_id_list : list = df["shop_id"].tolist()
    item_id_list : list = df["item_id"].tolist()
    item_cnt_day_list : list = df["item_cnt_day"].tolist()
    i : int = 0
    for date_block, shop_id, item_id, item_cnt in zip(data_block_num_list, shop_id_list, item_id_list, item_cnt_day_list):
        key = str(date_block) + "_" + str(shop_id) + "_" + str(item_id)
        if key in dict_new.keys():
            dict_new[key] += item_cnt
        else:
            dict_new[key] = item_cnt
    
    # make category dictionary
    df_item_id_cat_list : list = df_items["item_id"].tolist()
    df_cat_id_cat_list : list = df_items["item_category_id"].tolist()
    cat_dict : dict = {}
    item_id_cat : int
    cat_id : int
    for item_id_cat, cat_id in zip(df_item_id_cat_list,df_cat_id_cat_list):
        cat_dict[item_id_cat] = cat_id

    # create new dataframe
    i : int = 0
    id_list : list = []
    data_block_num_list : list = []
    shop_id_list : list = []
    item_id_list : list = []
    cat_id_list : list = []
    item_cnt_month_list : list = []
    lag_one_list : list = []
    lag_two_list : list = []
    lag_three_list : list = []
    lag_four_list : list = []
    key : str
    value : float
    for key, value in dict_new.items():
        # show progress
        if (i % 10000 == 0 and i > 0):
            print(i,"writing done") 
        data_block_num : str
        shop_id : str
        item_id : str 
        data_block_num, shop_id, item_id = key.split("_")
        id_list.append(i)
        data_block_num_list.append(int(data_block_num))
        shop_id_list.append(int(shop_id))
        item_id_list.append(int(item_id))
        cat_id_list.append(int(cat_dict[int(item_id)]))
        item_cnt_month_list.append(float(value))
        # Get Lag One
        lag_one_key : str = str(int(data_block_num) - 1) + "_" + shop_id + "_" + item_id
        if (lag_one_key in dict_new.keys()):
            lag_one_list.append(float(dict_new[lag_one_key]))
        else:
            lag_one_list.append(0)
        lag_two_key : str = str(int(data_block_num) - 2) + "_" + shop_id + "_" + item_id
        if (lag_two_key in dict_new.keys()):
            lag_two_list.append(float(dict_new[lag_two_key]))
        else:
            lag_two_list.append(0)
        lag_three_key : str = str(int(data_block_num) - 3) + "_" + shop_id + "_" + item_id
        if (lag_three_key in dict_new.keys()):
            lag_three_list.append(float(dict_new[lag_three_key]))
        else:
            lag_three_list.append(0)
        lag_four_key : str = str(int(data_block_num) - 4) + "_" + shop_id + "_" + item_id
        if (lag_four_key in dict_new.keys()):
            lag_four_list.append(float(dict_new[lag_four_key]))
        else:
            lag_four_list.append(0)
        i += 1

    df_new : DataFrame =  pd.DataFrame(
    {
        'id': id_list,
        'data_block_num': data_block_num_list,
        'shop_id': shop_id_list,
        'item_id': item_id_list,
        'cat_id': cat_id_list,
        'item_cnt_month': item_cnt_month_list,
        'lag_one': lag_one_list,
        'lag_two': lag_two_list,
        'lag_three': lag_three_list,
        'lag_four': lag_four_list
    })
    # save new dataframe
    df_new.to_csv(new_csv_file_name, index=None)  

create_test_csv("Data\\test.csv","Data\\items.csv","Data\\sales_train_processed.csv","Data\\test_processed.csv")               
convert_to_month_format("Data\\sales_train.csv","Data\\items.csv","Data\\sales_train_processed.csv")
#add_ID_field("Data\\sales_train.csv","Data\\sales_train_ID")