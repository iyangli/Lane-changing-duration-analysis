import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import math
import matplotlib.ticker as ticker
import mpl_toolkits.axisartist.axislines as axislines
from  pandas import  DataFrame as df
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 25)

import matplotlib.pyplot as plt
# 支持中文
font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
from scipy.stats import norm #使用直方图和最大似然高斯分布拟合绘制分布
plt.rcParams["axes.labelsize"] = 18

ori_data=pd.read_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20200922,first head of the LC data(threhold=1.5).xlsx",sheet_name="Sheet1")


ori_data["LCD(s)"]=ori_data["LCD"]
ori_data["Initial speed(m/s)"]=abs(ori_data["s v x"])
ori_data["Initial speed difference with preceding vehicle(m/s)"]=(ori_data["spediff p x"])
ori_data["Initial speed difference with target preceding vehicle(m/s)"]=(ori_data["spediff tp x"])
ori_data["Initial distance with preceding vehicle(m)"]=(ori_data["disdiff p"])
ori_data["Initial gap distance in the target lane(m)"]=(ori_data["disdiff p"])




plt.figure(figsize=(8, 6), dpi=600)
sns.jointplot(x='Initial gap distance in the target lane(m)', y='LCD(s)',data=ori_data,space = 0.2,ratio = 5,kind="hex", height = 8,
              marginal_kws=dict(bins=40,kde=True))
plt.tick_params(labelsize=16) #刻度字体大小13

plt.savefig('D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\pairplot dis tpf lcd.png',bbox_inches='tight',dpi=600)


plt.figure(figsize=(8, 6), dpi=600)
sns.jointplot(x='Initial speed(m/s)', y='LCD(s)',data=ori_data,space = 0.2,ratio = 5,kind="hex", height = 8,
              marginal_kws=dict(bins=40,kde=True))
plt.tick_params(labelsize=16) #刻度字体大小13

plt.savefig('D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\pairplot speed lcd.png',bbox_inches='tight',dpi=600)


plt.figure(figsize=(8, 6), dpi=600)
sns.jointplot(x='Initial speed difference with preceding vehicle(m/s)', y='LCD(s)',data=ori_data,space = 0.2,ratio = 5,kind="hex", height = 8,
              marginal_kws=dict(bins=40,kde=True))
plt.tick_params(labelsize=16) #刻度字体大小13

plt.savefig('D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\pairplot spediff p lcd.png',bbox_inches='tight',dpi=600)


plt.figure(figsize=(8, 6), dpi=600)
sns.jointplot(x='Initial speed difference with target preceding vehicle(m/s)', y='LCD(s)',data=ori_data,space = 0.2,ratio = 5,kind="hex", height = 8,
              marginal_kws=dict(bins=40,kde=True),)
plt.tick_params(labelsize=16) #刻度字体大小13

plt.savefig('D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\pairplot spediff tp lcd.png',bbox_inches='tight',dpi=600)


# plt.figure(figsize=(5, 5), dpi=600)
# sns.jointplot(x='Initial distance with preceding vehicle(m)', y='LCD(s)',data=ori_data,space = 0.2,ratio = 5,kind="hex", size = 8,
#               marginal_kws=dict(bins=40,kde=True),)
# plt.savefig('D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\pairplot disdiff p lcd.png',bbox_inches='tight',dpi=600)


