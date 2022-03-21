import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import numpy as np
import math
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)
import seaborn as sns
from lifelines import (GeneralizedGammaFitter, ExponentialFitter,
LogNormalFitter, LogLogisticFitter, NelsonAalenFitter,
PiecewiseExponentialFitter, GeneralizedGammaFitter, SplineFitter)
from lifelines.statistics import multivariate_logrank_test,pairwise_logrank_test
from lifelines.utils import restricted_mean_survival_time
from lifelines.plotting import rmst_plot
from lifelines.plotting import rmst_plot
from lifelines.utils import restricted_mean_survival_time
from lifelines.utils import median_survival_times
from lifelines import GeneralizedGammaFitter
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}
ori_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\LC_Extraction\\20210524,total data(together).xlsx")
ori_data["relative speed"]=ori_data["s v x"]-ori_data["p v x"]

data_vehicle=ori_data[ori_data["class"]=="vehicle"]
data_truck=ori_data[ori_data["class"]=="truck"]


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# dhw
data_vehicle["dhw group"]=pd.cut(data_vehicle["dhw"],[0,100,200,300,400],labels=["0~50m","50~100m","100~150m","150~200m"])
data_vehicle_dhw_group_1=data_vehicle[data_vehicle["dhw group"]=="0~50m"]
data_vehicle_dhw_group_2=data_vehicle[data_vehicle["dhw group"]=="50~100m"]
data_vehicle_dhw_group_3=data_vehicle[data_vehicle["dhw group"]=="100~150m"]
data_vehicle_dhw_group_4=data_vehicle[data_vehicle["dhw group"]=="150~200m"]
gg_data_vehicle_dhw_group_1 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_1["LCD"], data_vehicle_dhw_group_1["complete"], label='0~50m(vehicle)')
gg_data_vehicle_dhw_group_2 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_2["LCD"], data_vehicle_dhw_group_2["complete"], label='50~100m(vehicle)')
gg_data_vehicle_dhw_group_3 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_3["LCD"], data_vehicle_dhw_group_3["complete"], label='100~150m(vehicle)')
# gg_data_vehicle_dhw_group_4 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_4["LCD"], data_vehicle_dhw_group_4["complete"], label='150~200m(vehicle)')
data_truck["dhw group"]=pd.cut(data_truck["dhw"],[0,100,200,300,400],labels=["0~50m","50~100m","100~150m","150~200m"])
data_truck_dhw_group_1=data_truck[data_truck["dhw group"]=="0~50m"]
data_truck_dhw_group_2=data_truck[data_truck["dhw group"]=="50~100m"]
data_truck_dhw_group_3=data_truck[data_truck["dhw group"]=="100~150m"]
data_truck_dhw_group_4=data_truck[data_truck["dhw group"]=="150~200m"]
gg_data_truck_dhw_group_1 = GeneralizedGammaFitter().fit(data_truck_dhw_group_1["LCD"], data_truck_dhw_group_1["complete"], label='0~50m(truck)')
gg_data_truck_dhw_group_2 = GeneralizedGammaFitter().fit(data_truck_dhw_group_2["LCD"], data_truck_dhw_group_2["complete"], label='50~100m(truck)')
gg_data_truck_dhw_group_3 = GeneralizedGammaFitter().fit(data_truck_dhw_group_3["LCD"], data_truck_dhw_group_3["complete"], label='100~150m(truck)')
# gg_data_truck_dhw_group_4 = GeneralizedGammaFitter().fit(data_truck_dhw_group_4["LCD"], data_truck_dhw_group_4["complete"], label='150~200m(truck)')
print("小汽车dhw分组")
print(gg_data_vehicle_dhw_group_1.median_survival_time_)
print(gg_data_vehicle_dhw_group_2.median_survival_time_)
print(gg_data_vehicle_dhw_group_3.median_survival_time_)
# print(gg_data_vehicle_dhw_group_4.median_survival_time_)
print("卡车dhw分组")
print(gg_data_truck_dhw_group_1.median_survival_time_)
print(gg_data_truck_dhw_group_2.median_survival_time_)
print(gg_data_truck_dhw_group_3.median_survival_time_)
# print(gg_data_truck_dhw_group_4.median_survival_time_)



plt.figure(figsize=(10,10))
ax1 = plt.subplot(111)
gg_data_vehicle_dhw_group_1.plot_survival_function(ci_show=False)
gg_data_vehicle_dhw_group_2.plot_survival_function(ci_show=False)
gg_data_vehicle_dhw_group_3.plot_survival_function(ci_show=False)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(a) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("survival function(vehicle)",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.show()








