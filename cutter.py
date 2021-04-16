"""
Cut a dataframe by item number to a ratio for train validation split
"""


from pandas.core.frame import DataFrame


class data_splitter():
    @staticmethod
    def split(ratio : float, dataframe : DataFrame):
        """
        split by item number 
        """
        dataframe.sort_values
