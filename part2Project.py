import pandas as pd
from pandas import DataFrame
import numpy as np

gdp_data = pd.read_excel("GDP_Merged_Dataset.xlsx",sheet_name="Merged Data")
gdp_data = DataFrame(data=gdp_data)

years = gdp_data['Year']
print(years)
convert_col = ['Personal income','Wages and salaries','Compensation of employees','Gross domestic product',
               'Personal consumption expenditures','Disposable personal income']
for col in convert_col:
    gdp_data[col + " Year Over Year"] = gdp_data[col].pct_change(1).round(3)

print(gdp_data.columns)
gdp_data['Personal Savings rate'] = (gdp_data['Personal saving as a percentage of disposable personal income']).round(3)
gdp_data['Wage Ratio'] = (gdp_data['Wages and salaries Year Over Year']/ gdp_data['Personal income Year Over Year']).round(3)
gdp_data['Transfer Dependency'] = (gdp_data['Personal current transfer receipts'] / gdp_data['Personal income']).round(3)
gdp_data['Personal consumption expenditures Shares'] = (gdp_data['Personal consumption expenditures'] / gdp_data['Gross domestic product']).round(3)
gdp_data['Tax Ratio'] = (gdp_data['Personal current taxes'] / gdp_data['Personal income']).round(3)

gdp_data.drop(columns=convert_col,inplace=True)
gdp_data.drop(columns=['Personal saving','Personal current transfer receipts','Personal current taxes'],inplace=True)

print(gdp_data.to_string())



