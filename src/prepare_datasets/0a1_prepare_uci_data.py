# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Download UCI Data
# Here, I will read in data from the UCI repository and do some minor preprocessing.

# %%
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from leakconfound.prepare_datasets import (dump_data,
                                           balance_shuffle_data,
                                           encode_cat)


np.random.seed(4224)

data_dir = sys.argv[1]

# %%
Path(f'{data_dir}/uci_datasets/').mkdir(exist_ok=True, parents=True)

conf_path = f'{data_dir}/uci_adj_lists/'
Path(conf_path).mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Helper Functions Download data

# %% [markdown]
# # Classification Datasets

# %% [markdown]
# ## Adult/Income Dataset
#
# [more info]('https://archive.ics.uci.edu/ml/datasets/Adult')
#
# * Target: <=50K
#     Does a person gain less/equal to 50k a year
# * Confound:
# %%

# infered from the UCI page
names = [
    'age__continuous',
    'workclass__categorical',
    'fnlwgt__continuous',
    'education__categorical',
    'education-num__continuous',
    'marital-status__categorical',
    'occupation__categorical',
    'relationship__categorical',
    'race__categorical',
    'sex__categorical',
    'capital-gain__continuous',
    'capital-loss__continuous',
    'hours-per-week__continuous',
    'native-country__categorical',
    'earnings__binary_target'
]


# %%
df_income = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    header=None, names=names)
df_income = balance_shuffle_data(df_income)
df_income, income_cat_encoders = encode_cat(df_income)
dump_data(df_income, income_cat_encoders, 'income', f'{data_dir}/')
df_income.head()

# %% [markdown]
# ##  Bank Marketing Dataset
#
# [more info](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

# %%
bank_names = [
    'age__continuous',
    'job__categorical',
    'marital__categorical',
    'education__categorical',
    'default__categorical',
    'housing__categorical',
    'loan__categorical',
    'contract__categorical',
    'month__categorical',
    'day_of_week__categorical',
    'duration__continuous',
    'campaign__continuous',
    'pdays__continuous',
    'previous__continuous',
    'poutcome__categorical',
    'emp_var_rate__continuous',
    'cons_price_idx__continuous',
    'cons_conf_idx__continuous',
    'euribor3m__continuous',
    'nr_employed__continuous',
    'term_deposition__binary_target'
]

# %%
df_bank = pd.read_csv(
    f'{data_dir}/raw/bank-additional/bank-additional-full.csv', sep=';')
df_bank.head()

# %%
# check renamed vs orignal
[(name.split('__')[0].replace('_', '.'), col_name)
 for name, col_name in zip(bank_names, df_bank.columns.to_list())]


# %%
df_bank.columns = bank_names

# dropping because of too many missing values encoded as 999
df_bank = df_bank.drop(columns=["pdays__continuous"])
df_bank = balance_shuffle_data(df_bank)
df_bank, bank_cat_encoders = encode_cat(df_bank)
dump_data(df_bank, bank_cat_encoders, 'bank', f'{data_dir}/')

# %% [markdown]
# ## Heart Disease Data Set
#
# [more info](https://archive.ics.uci.edu/ml/datasets/heart+disease)
#

# %%
names_heart = [
    'age__continuous',
    'sex__categorical',
    'chest_pain__categorical',
    'resting_blood_pressure__continuous',
    'serum_cholestoral__continuous',
    'fasting_blood_sugar__categorical',
    'restecg__categorical',
    'thalach__continious',
    'exang__categorical',
    'oldpeak__continious',
    'slope__categorical',
    'ca__continuous',  # number of major vessels 0-3
    'thal__categorical',
    'num__multiclass_target'
]

# %%
df_heart = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/heart-disease/processed.cleveland.data',
    header=None, names=names_heart)

# %%

# missing values as np.na and then dropped
df_heart = df_heart.applymap(lambda val: np.nan if val == '?' else val)
df_heart = df_heart.dropna()

# binarize target as proposed for this dataset in UCI repository
df_heart['num__binary_target'] = (df_heart['num__multiclass_target']
                                  .apply(lambda x: x if x == 0 else 1)
                                  )
df_heart.drop(columns=['num__multiclass_target'], inplace=True)

# %%
df_heart = balance_shuffle_data(df_heart)
df_heart, heart_cat_encoders = encode_cat(df_heart)
dump_data(df_heart, heart_cat_encoders, 'heart', f'{data_dir}/')

# %% [markdown]
# ## Blood Transfusion
# [more_info](https://archive.ics.uci.edu/ml/datasets/Blood+Transfusion+Service+Center)

names = ['recency__continuous', 'frequency__continuous',
         'monetary__continuous', 'time__continuous', 'donated__binary_target']

df_blood = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data')
df_blood.columns = names
df_blood.head()

# %%
df_blood = balance_shuffle_data(df_blood)
df_blood, blood_cat_encoders = encode_cat(df_blood)
dump_data(df_blood, blood_cat_encoders, 'blood', f'{data_dir}/')

# %% [markdown]
# ## Breast Cancer Data Sets
# [more info](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

# %% [markdown]
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry
# j) fractal dimension ("coastline approximation" - 1)

# %%
names = ['radius', 'texture',
         'perimeter', 'area',
         'smoothness', 'compactness',
         'concavity', 'concave_points',
         'symmetry', 'fractal_dimension']

columns = ['id', 'diagnosis__binary_target']
columns += [name + measure for measure in ['_mean__continuous',
                                           '_std__continuous', '_worst__continuous']
            for name in names]


# %%
df_cancer = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
    header=None)
df_cancer.columns = columns

# %%
df_cancer.drop(columns='id', inplace=True)
df_cancer = balance_shuffle_data(df_cancer)
df_cancer, cancer_cat_encoders = encode_cat(df_cancer)
dump_data(df_cancer, cancer_cat_encoders, 'cancer', f'{data_dir}/')

# %% [markdown]
# # Regression Data Sets

# %% [markdown]
# ## Student Performance
#
# [more info](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

# %%
names_student = [
    'school__categorical',
    'sex__categorical',
    'age__continuous',
    'address__categorical',
    'famsize__categorical',
    'Pstatus__categorical',
    'Medu__categorical',
    'Fedu__categorical',
    'Mjob__categorical',
    'Fjob__categorical',
    'reason__categorical',
    'guardian__categorical',
    'traveltime__categorical',
    'studytime__categorical',
    'failures__categorical',
    'schoolsup__categorical',
    'famsup__categorical',
    'paid__categorical',
    'activities__categorical',
    'nursery__categorical',
    'higher__categorical',
    'internet__categorical',
    'romantic__categorical',
    'famrel__categorical',
    'freetime__categorical',
    'goout__categorical',
    'Dalc__categorical',
    'Walc__categorical',
    'health__categorical',
    'absences__continuous',
    'G1__continuous',
    'G2__continuous',
    'G3__regression_target'
]

# %%
df_student = pd.read_csv(f'{data_dir}/raw/student/student-mat.csv', sep=';')

# check renamed vs orignal
[(name.split('__')[0].replace('_', '.'), col_name)
 for name, col_name in zip(names_student, df_student.columns.to_list())]


# %%
df_student.columns = names_student
df_student.drop(columns=['G1__continuous', 'G2__continuous'],
                inplace=True)  # to make the prediction a bit harder
df_student.head()

# %%
df_student = balance_shuffle_data(df_student)
df_student, student_cat_encoders = encode_cat(df_student)
dump_data(df_student, student_cat_encoders, 'student', f'{data_dir}/')

# %% [markdown]
# ## Abalone Data Set
#
# [more info](https://archive.ics.uci.edu/ml/datasets/Abalone)
#
#
# From documentation on UCi website:
# "Given is the attribute name, attribute type, the measurement unit and a brief descriptionself.
# The number of rings is the value to predict:
# either as a continuous value or as a classification problem. "
#
# Here, we will chose continuous

# %%
columns = ['Sex__categorical',
           'Length__continuous',
           'Diameter__continuous',
           'Height__continuous',
           'Whole_weight__continuous',
           'Shucked_weight__continuous',
           'Viscera_weight__continuous',
           'Shell_weight__continuous',
           'Ring__regression_target']

# %%
df_abalone = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data')
df_abalone.columns = columns

# %%
df_abalone = balance_shuffle_data(df_abalone)
df_abalone, abalone_cat_encoders = encode_cat(df_abalone)
dump_data(df_abalone, abalone_cat_encoders, 'abalone', f'{data_dir}/')

# %% [markdown]
# ## Concrete Dat set
# [more info](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength)

# %%
columns = ['cement__continuous',
           'blast_furnace_slag__continuous',
           'fly_ash__continuous',
           'water__continuous',
           'superplasticizer__continuous',
           'coarse_aggregate__continuous',
           'fine_aggregate__continuous',
           'age__continuous',
           'concrete_compressive_strength__regression_target']

df_concrete = pd.read_excel(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/concrete/compressive/Concrete_Data.xls')
df_concrete.columns = columns

# %%
df_concrete.head()

# %%
df_concrete = balance_shuffle_data(df_concrete)
df_concrete, concrete_cat_encoders = encode_cat(df_concrete)
dump_data(df_concrete, concrete_cat_encoders, 'concrete', f'{data_dir}/')

# %% [markdown]
# ## Residential Building Data Set
#

# %%
df_building = pd.read_excel(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases/00437/Residential-Building-Data-Set.xlsx')
df_building.head()


# %%
# Renaming the columns

# Getting the outer heading for each col
previous_col = df_building.columns[0]
all_columns = [previous_col]

for col in df_building.columns[1:]:
    if not col.startswith('Unnamed'):
        previous_col = col
    all_columns.append(previous_col)

columns = (pd.Series(all_columns)
           .str.replace(r'PROJECT DATES \(PERSIAN CALENDAR\)', 'dates')
           .str.replace('PROJECT PHYSICAL AND FINANCIAL VARIABLES', 'physic_finance')
           .str.replace('ECONOMIC VARIABLES AND INDICES IN TIME LAG ', 'eco_')

           )

# getting the 2nd / sub heading
sub_cols = (df_building
            .iloc[0, :].reset_index(drop=True)
            .str.lower()
                .str.replace('-| ', '_')
            )

# every fature is continious except for the quarters

column_type = ['__continuous'] * len(columns)
column_type[1] = '__categorical'
column_type[3] = '__categorical'


columns += '_' + sub_cols + column_type
df_building = df_building.iloc[1:, :]

# construction cost 2nd target
# price it sells for is the actual target
columns.iloc[-2] = 'construction_cost__regression_target'
columns.iloc[-1] = 'sales_price__regression_target'

df_building.columns = columns
df_building.drop(
    columns=['construction_cost__regression_target'], inplace=True)
# removing project dates as they are not mentioned as features in description
df_building.iloc[:, 4:]
df_building.head()

# %%
df_building = balance_shuffle_data(df_building)
df_building, building_cat_encoders = encode_cat(df_building)
dump_data(df_building, building_cat_encoders, 'building', f'{data_dir}/')

# %% [markdown]
#

# %% [markdown]
# ## Real Estate
# [more_info](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set)

# %%
df_real_estate = pd.read_excel(
    r'https://archive.ics.uci.edu/ml/machine-learning-databases'
    r'/00477/Real%20estate%20valuation%20data%20set.xlsx')
df_real_estate.head()

# %%
df_real_estate = df_real_estate.drop(columns=['No', 'X1 transaction date'])
columns = (df_real_estate.columns.to_series()
           .str.replace('X[0-9] ', '')
           .str.replace(' ', '_')
           )
columns += '__continuous'
columns[-1] = 'house_price_of_unit_area__regression_target'

df_real_estate.columns = columns
df_real_estate.head()

# %%
df_real_estate = balance_shuffle_data(df_real_estate)
df_real_estate, real_estate_cat_encoders = encode_cat(df_real_estate)
dump_data(df_real_estate, real_estate_cat_encoders,
          'real_estate', f'{data_dir}/')
