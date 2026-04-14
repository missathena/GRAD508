import confusion_matrix
import numpy as np
import pandas as pd
import pca as pca
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pandas import DataFrame
from sklearn.utils.multiclass import type_of_target

#Pulling and combining data
school_head = pd.read_csv('School head questionnaire data_REDS_UAE.csv')
school_head = DataFrame(school_head)
print(f"School head questionnaire: {school_head.columns}")

student_questionnaire = pd.read_csv('Student questionnaire data_REDS_UAE.csv')
student_questionnaire = DataFrame(student_questionnaire)
print(f"Student questionnaire: {student_questionnaire.columns}")

teacher_questionnaire = pd.read_csv('Teacher questionnaire data_REDS_UAE.csv')
teacher_questionnaire = DataFrame(teacher_questionnaire)
print(f"Teacher questionnaire: {teacher_questionnaire.columns}")

school_ID =DataFrame(school_head['IDSCHOOL'].drop_duplicates())
#print(school_ID.to_string())
student_id = DataFrame(student_questionnaire['IDSTUD'].drop_duplicates())
student_id_school_id = DataFrame(student_questionnaire['IDSCHOOL'].drop_duplicates())

teacher_school_id = DataFrame(teacher_questionnaire['IDSCHOOL'].drop_duplicates())

merged_school_ids = pd.merge(student_id_school_id, teacher_school_id, on='IDSCHOOL')
merged_school_ids = pd.merge(merged_school_ids, school_ID, on='IDSCHOOL')
merged_school_ids = set(merged_school_ids['IDSCHOOL'])
#print(merged_school_ids)

independent_work = teacher_questionnaire[['IDSCHOOL','IT06D_indep']]
time_spent = teacher_questionnaire[['IDSCHOOL','IT05F_indiv']]

independent_work = independent_work.groupby('IDSCHOOL').mean()
time_spent = time_spent.groupby('IDSCHOOL').mean()

teacher_agg = pd.merge(time_spent, independent_work, on='IDSCHOOL')


merged_questionnaires = pd.merge(student_questionnaire, teacher_agg,on='IDSCHOOL')
merged_questionnaires = pd.merge(merged_questionnaires,school_head,on='IDSCHOOL')

#print(merged_questionnaires.describe()..round(2).to_string())

#Correlation matrix
corr_matrix = merged_questionnaires[['TOTWGTS', 'IS18C_plan_capacity', 'ASDAGE',
       'SEX_female', 'IS34_lang', 'prnt_ed_M', 'IS17G_miss_meals',
       'IS01_prop_sch_time', 'IS09_comm_modes', 'IS21_teach_sup',
       'IS18E_adult_help', 'IS14A', 'IS14C', 'IS14K', 'IS17A', 'IS17C',
       'IS17E', 'IS25A', 'IS25B', 'IS25C', 'IS25D', 'IS25E', 'IS25F', 'IS25G',
       'IS25H', 'IS25J', 'IT05F_indiv', 'IT06D_indep', 'IP13D_prnt_sup',
       'IP21_remote_prep', 'IP34_locale_size', 'IP34A_public',
       'IP1G35C_prop_disadvan', 'IP03_tech']].corr(method='pearson').stack().to_frame(name='correlation').round(2)

#print(merged_questionnaires.isnull().sum().to_dict())

extent_of_difficulty = merged_questionnaires[['IS14A', 'IS14C', 'IS14K', 'IS17A', 'IS17C',
       'IS17E', 'IS25A', 'IS25B', 'IS25C', 'IS25D', 'IS25E', 'IS25F', 'IS25G',
       'IS25H', 'IS25J']]
extent_of_difficulty.fillna(0, inplace=True)
#print(extent_of_difficulty.columns)
#sns.heatmap(extent_of_difficulty.corr())
#PCA using pca
'''x = extent_of_difficulty
analysis = pca(n_components=0.85,normalize=True,detect_outliers=['ht2'], alpha=0.05)
diff_pca = analysis.fit_transform(x)
#print(diff_pca['loadings'])
#print(diff_pca['outliers'])
outlier = diff_pca['outliers']'''

#PCA using sklearn
x = extent_of_difficulty
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=0.85)
x_pca = pca.fit_transform(x_scaled)
x_pca = pd.DataFrame(data=x_pca,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'])

print(pca.explained_variance_ratio_.round(3))
#print(pca.explained_variance_ratio_.cumsum().round(3))

#Decision Tree for feature selection
plan_capacity = merged_questionnaires['IS18C_plan_capacity']
plan_capacity.fillna(0, inplace=True)

y = plan_capacity
X_train, X_test, y_train, y_test = train_test_split(x, y)

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)

clf = DecisionTreeClassifier()
clf.fit(x_train_scaled, y_train)

importances =  clf.feature_importances_
threshold = .08
selected_features = x.columns[importances > threshold]

#Prediction Model
used_selected_features = ['IS17A', 'IS17C', 'IS17E']
x_train_selected = X_train[used_selected_features]
x_test_selected = X_test[used_selected_features]

mlr = LogisticRegression(class_weight='balanced',solver='lbfgs')
mlr.fit(x_train_selected, y_train)
y_pred = mlr.predict(x_test_selected)
y_proba = mlr.predict_proba(x_test_selected)
print(mlr.classes_)

test_accuracy = mlr.score(x_test_selected, y_test)
train_accuracy = mlr.score(x_train_selected, y_train)

print(test_accuracy, train_accuracy)
print(classification_report(y_test, y_pred,target_names=['IS18C_plan_capacity','IS17A', 'IS17C', 'IS17E']))
print(y_pred)