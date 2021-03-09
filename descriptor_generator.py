# -*- coding: utf-8 -*-
"""
Created on Fri May  9 23:00:41 2020
@author: Zhenzhang Li
"""

# import general python package/ read in compounds list
import pandas as pd

df = pd.read_excel(r'c_pounds.xls')
df.head()
df.dtypes
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
from statistics import mean


class Vectorize_Formula:

    def __init__(self):
        elem_dict = pd.read_excel(r'elements.xls')  # CHECK NAME OF FILE，前边的r是对斜杠进行转义
        self.element_df = pd.DataFrame(elem_dict)
        self.element_df.set_index('Symbol', inplace=True)
        self.column_names = []
        for string in ['avg', 'diff', 'max', 'min', 'std']:
            for column_name in list(self.element_df.columns.values):
                self.column_names.append(string + '_' + column_name)

    def get_features(self, formula):
        try:
            fractional_composition = mg.Composition(formula).fractional_composition.as_dict()#显示化学式归一成分
            element_composition = mg.Composition(formula).element_composition.as_dict()#显示化学式成分
            avg_feature = np.zeros(len(self.element_df.iloc[0]))
            std_feature = np.zeros(len(self.element_df.iloc[0]))
            for key in fractional_composition:
                try:
                    avg_feature += self.element_df.loc[key].values * fractional_composition[key]
                    #element_df.loc[key].values为元素表中相应属性值，
                    #fractional_composition[key]为化学成分中元素所对应的成分
                    #element_df.loc[key].values * fractional_composition[key]=原子属性值在化学式式中所占有的比例值
                    diff_feature = self.element_df.loc[list(fractional_composition.keys())].max() - self.element_df.loc[
                        list(fractional_composition.keys())].min()
                    #找出化学式中每种原子的每种属性的最大值和最小值，然后相减
                except Exception as e:
                    print('The element:', key, 'from formula', formula, 'is not currently supported in our database')
                    return np.array([np.nan] * len(self.element_df.iloc[0]) * 5)
            max_feature = self.element_df.loc[list(fractional_composition.keys())].max()
            min_feature = self.element_df.loc[list(fractional_composition.keys())].min()
            std_feature = self.element_df.loc[list(fractional_composition.keys())].std(ddof=0)
            # 把相关的信息拼接成
            features = pd.DataFrame(np.concatenate(
                [avg_feature, diff_feature, np.array(max_feature), np.array(min_feature), np.array(std_feature)]))
            features = np.concatenate(
                [avg_feature, diff_feature, np.array(max_feature), np.array(min_feature), np.array(std_feature)])
            return features.transpose()
        except:
            print(
                'There was an error with the Formula: ' + formula + ', this is a general exception with an unkown error')
            return [np.nan] * len(self.element_df.iloc[0]) * 5


gf = Vectorize_Formula()

# empty list for storage of features
features = []

# add values to list using for loop
for formula in df['Composition']:
    features.append(gf.get_features(formula))

# feature vectors and targets as X and y
X = pd.DataFrame(features, columns=gf.column_names)
pd.set_option('display.max_columns', None)
# allows for the export of data to excel file
header = gf.column_names
header.insert(0, "Composition")#在header第一列中插入Composition这列

composition = pd.read_excel('c_pounds.xls', sheet_name='Sheet1', usecols="A")
#读取c_pounds.xlsx表中的第一列
composition = pd.DataFrame(composition)

predicted = np.column_stack((composition, X))#按照列拼接
predicted = pd.DataFrame(predicted)
predicted.to_excel('to_predict_relative_permittivity.xls', index=False, header=header)#保存为.xlsx文件
print("A file named to_predict_relative_permittivity.xls has been generated.\nPlease check your folder.")