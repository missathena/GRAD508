import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.tree import plot_tree

gdp_data = pd.read_excel("GDP_Merged_Dataset.xlsx", sheet_name="Merged Data")
gdp_data = DataFrame(data=gdp_data)

# creating year over year ratios and others
convert_col = ['Personal income', 'Wages and salaries', 'Compensation of employees', 'Gross domestic product',
               'Personal consumption expenditures', 'Disposable personal income']
for col in convert_col:
    gdp_data[col + " Year Over Year"] = gdp_data[col].pct_change(1).round(3)

gdp_data['Personal Savings rate'] = (
        (gdp_data['Wages and salaries'] - gdp_data['Personal consumption expenditures']) / gdp_data[
    'Wages and salaries']).round(3)
gdp_data['Wage Ratio'] = (
        gdp_data['Wages and salaries'] / gdp_data['Personal income']).round(3)
gdp_data['Transfer Dependency'] = (gdp_data['Personal current transfer receipts'] / gdp_data['Personal income']).round(
    3)
gdp_data['Personal consumption expenditures Shares'] = (
        gdp_data['Personal consumption expenditures'] / gdp_data['Gross domestic product']).round(3)
gdp_data['Tax Ratio'] = (gdp_data['Personal current taxes'] / gdp_data['Personal income']).round(3)

gdp_data.drop(columns=convert_col, inplace=True)
gdp_data.drop(columns=['Personal saving', 'Personal current transfer receipts', 'Personal current taxes'], inplace=True)

# add lag
lag_col = ['Wages and salaries Year Over Year', 'Disposable personal income Year Over Year',
           'Personal consumption expenditures Year Over Year', 'Personal Savings rate', 'Transfer Dependency']
for col in lag_col:
    gdp_data[col + ' lag1'] = gdp_data[col].shift(1)
    gdp_data[col + ' lag2'] = gdp_data[col].shift(2)

# rolling statistics
rolling_stats = ['Wages and salaries Year Over Year', 'Personal consumption expenditures Year Over Year',
                 'Gross domestic product Year Over Year']

for col in rolling_stats:
    gdp_data[col + ' rmean3'] = gdp_data[col].rolling(window=3, min_periods=2).mean().round(3)
    gdp_data[col + ' rstd3'] = gdp_data[col].rolling(window=3, min_periods=2).std().round(3)
    gdp_data[col + ' trend_dev'] = gdp_data[col] - gdp_data[col + ' rmean3']

# recession indicator
gdp_data['is recession'] = (gdp_data['Gross domestic product Year Over Year'] < 0).astype(int)

# data cleaning
gdp_data['prediction_target'] = gdp_data['Personal consumption expenditures Year Over Year'].shift(-1)
gdp_data.dropna(inplace=True)
gdp_data.drop(columns=['Supplements to wages and salaries',
                       'Rental income of persons with capital consumption adjustment',
                       'Personal interest income', 'Personal dividend income',
                       'Government social benefits to persons', 'Social security', 'Medicare',
                       'Medicaid', 'Unemployment insurance',
                       'Personal saving as a percentage of disposable personal income',
                       'Personal consumption expenditures.1', 'Durable goods',
                       'Nondurable goods', 'Services', 'Gross private domestic investment',
                       'Fixed investment',
                       'Government consumption expenditures and gross investment',
                       'Proprietors\' income with inventory valuation and capital consumption adjustments'],
              inplace=True)

gdp_data.to_csv("GDP_Merged_Dataset.csv")

# train data and test data split
gdp_data['prediction_target'] = gdp_data['Personal consumption expenditures Year Over Year'].shift(-1)
gdp_data.dropna(subset=['prediction_target'], inplace=True)

x = gdp_data.drop(columns=['prediction_target'])
y = gdp_data['prediction_target']

time_series_split = TimeSeriesSplit(n_splits=5)

for train_index, test_index in time_series_split.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = RobustScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

#Random Forest Regression
random_forest = RandomForestRegressor(n_estimators=500, random_state=42, max_depth=10, max_features=15,
                                      min_samples_leaf=3,oob_score=True)
random_forest.fit(x_train_scaled, y_train)
y_pred = random_forest.predict(x_test_scaled)

feature_importance = pd.DataFrame(zip(gdp_data.columns,random_forest.feature_importances_))
feature_name = gdp_data.columns

r_mean_sq_err = (mean_squared_error(y_test, y_pred)).__round__(3)
r2 = (r2_score(y_test, y_pred)).__round__(3)
out_of_bag = (random_forest.oob_score_).__round__(3)
root_mean_sq_err = (mean_squared_error(y_test, y_pred)).__round__(3)
explained_var = (explained_variance_score(y_test, y_pred)).__round__(3)

#print(feature_name)
print(f"Out of Bag Score: {out_of_bag}")
print(f"r squared score: {r2}")
print(f"Explained Variance score: {explained_var}")
print(f"Root Mean Squared Error: {root_mean_sq_err}")
print(f"Mean squared error: {r_mean_sq_err}")

tree = random_forest.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(tree,feature_names=feature_name,filled=True,max_depth=10)
plt.savefig(fname="tree.png")

