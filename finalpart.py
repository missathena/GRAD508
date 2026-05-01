import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

gdp_data = pd.read_excel("GDP_Merged_Dataset.xlsx", sheet_name="Merged Data")
gdp_data = DataFrame(data=gdp_data)
gdp_no_years = gdp_data.drop(columns=["Year"])

#transform to logged data
gdp_no_years_growth = np.log(gdp_no_years).diff().round(4)
gdp_no_years_growth = gdp_no_years_growth.dropna()

#print(gdp_no_years_growth.head())

