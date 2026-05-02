import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.tree import plot_tree

# load data
gdp_data = pd.read_csv("GDP_Merged_Dataset-Merged Data.csv")

# remove commas set index as years
for col in gdp_data.columns:
    if col != "Year":
        gdp_data[col] = (gdp_data[col].astype(str).str.replace(",", "").str.strip().astype(float))
gdp_data = gdp_data.set_index("Year")

# gpd is target
gdp_data['prediction_target'] = np.log(gdp_data['Gross domestic product']).diff().shift(-1)

# creating year over year ratios and others
convert_col = ['Personal income', 'Personal consumption expenditures', 'Wages and salaries',
               'Compensation of employees', 'Gross domestic product', 'Disposable personal income']
for col in convert_col:
    gdp_data[col + " Year Over Year"] = gdp_data[col].pct_change()

gdp_data['Personal Savings rate'] = gdp_data['Personal saving as a percentage of disposable personal income'] / 100
gdp_data['Wage Ratio'] = (gdp_data['Wages and salaries'] / gdp_data['Personal income'])
gdp_data['Transfer Dependency'] = (gdp_data['Personal current transfer receipts'] / gdp_data['Personal income'])
gdp_data['Personal consumption expenditures Shares'] = (
        gdp_data['Personal consumption expenditures'] / gdp_data['Gross domestic product'])
gdp_data['Tax Ratio'] = (gdp_data['Personal current taxes'] / gdp_data['Personal income'])

gdp_data.drop(columns=convert_col, inplace=True)
gdp_data.drop(columns=['Personal saving', 'Personal current transfer receipts', 'Personal current taxes'], inplace=True)

# rolling statistics
rolling_stats = ['Wages and salaries Year Over Year', 'Personal consumption expenditures Year Over Year',
                 'Gross domestic product Year Over Year']

for col in rolling_stats:
    rmean = gdp_data[col].rolling(3, min_periods=2).mean().shift(1)
    gdp_data[col + ' rmean3'] = rmean
    gdp_data[col + ' rstd3'] = gdp_data[col].rolling(3, min_periods=2).std().shift(1)
    gdp_data[col + ' trend_dev'] = gdp_data[col] - rmean

# add lag
lag_col = ['Wages and salaries Year Over Year', 'Disposable personal income Year Over Year',
           'Personal consumption expenditures Year Over Year', 'Personal Savings rate', 'Transfer Dependency']
for col in lag_col:
    gdp_data[col + ' lag1'] = gdp_data[col].shift(1)
    gdp_data[col + ' lag2'] = gdp_data[col].shift(2)

# recession indicator + add lag
gdp_data['is recession'] = (gdp_data['Gross domestic product Year Over Year'] < 0).astype(int)
gdp_data['is recession lag1'] = gdp_data['is recession'].shift(1).fillna(0).astype(int)

# data cleaning
gdp_data = gdp_data.dropna(subset=['prediction_target'])
gdp_data.drop(columns=['Supplements to wages and salaries',
                       'Rental income of persons with capital consumption adjustment',
                       'Personal interest income', 'Personal dividend income',
                       'Government social benefits to persons', 'Social security', 'Medicare',
                       'Medicaid', 'Unemployment insurance',
                       'Personal saving as a percentage of disposable personal income', 'Durable goods',
                       'Nondurable goods', 'Services', 'Gross private domestic investment',
                       'Fixed investment', 'Personal consumption expenditures.1',
                       'Government consumption expenditures and gross investment',
                       'Proprietors\' income with inventory valuation and capital consumption adjustments'],
              inplace=True)
gdp_data.fillna(0, inplace=True)
gdp_data.to_csv('randomForest_dataset.csv')

# train data and test data split
x = gdp_data.drop(columns=['prediction_target'])
y = gdp_data['prediction_target']

x_train = x[x.index <= 2014]
x_test = x[x.index > 2014]
y_train = y[y.index <= 2014]
y_test = y[y.index > 2014]

# Random Forest Regression
random_forest = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=5, max_features='sqrt',
                                      min_samples_leaf=5, min_samples_split=5, oob_score=True, n_jobs=-1
                                      )
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

#feature importance
feature_name = x.columns
feature_importance = (
    pd.DataFrame({'feature': feature_name, 'importance': random_forest.feature_importances_})
    .sort_values('importance', ascending=False)
    .reset_index(drop=True)
)

print(feature_importance.head(15).to_string())

rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
mae  = round(mean_absolute_error(y_test, y_pred), 4)
r2   = round(r2_score(y_test, y_pred), 4)
explained_var = round(explained_variance_score(y_test, y_pred), 4)
oob  = round(random_forest.oob_score_, 3)

print(f"Out of Bag Score: {oob}")
print(f"R² Score: {r2}")
print(f"Explained Variance: {explained_var}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

#tree visualization
'''tree = random_forest.estimators_[0]
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_name, filled=True, max_depth=5)
plt.savefig("tree.png", dpi=150, bbox_inches='tight')'''

#residuals
residuals = pd.DataFrame({'Year': y_test.index,'Residual': y_test.values - y_pred})

fig, ax = plt.subplots(figsize=(11, 6))

ax.fill_between(residuals['Year'], residuals['Residual'], 0,
                where=(residuals['Residual'] >= 0),
                alpha=0.3, color='#2E86AB', label='Under-prediction')
ax.fill_between(residuals['Year'], residuals['Residual'], 0,
                where=(residuals['Residual'] < 0),
                alpha=0.3, color='#C73E1D', label='Over-prediction')

sns.lineplot(data=residuals, x='Year', y='Residual',
             color='#1B3A57', lw=2.2, marker='o', markersize=9,
             markerfacecolor='white', markeredgewidth=2, ax=ax)

ax.axhline(0, ls='--', c='black', lw=1.2, alpha=0.7)

for _, row in residuals.iterrows():
    offset = 0.0008 if row['Residual'] >= 0 else -0.0012
    ax.text(row['Year'], row['Residual'] + offset, f"{row['Residual']:+.4f}",
            ha='center', fontsize=9, fontweight='bold',
            color='#1B3A57')

ax.set_title("Residuals Random Forest GDP Growth Model",
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Year", fontsize=11, fontweight='bold')
ax.set_ylabel("Residual  (Actual − Predicted)", fontsize=11, fontweight='bold')
ax.set_xticks(residuals['Year'])
ax.legend(loc='best', frameon=True)
plt.tight_layout()
plt.show()