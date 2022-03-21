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


ori_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\LC_Extraction\\20210524,total data(together).xlsx")
ori_data["relative speed"]=ori_data["s v x"]-ori_data["p v x"]

data_vehicle=ori_data[ori_data["class"]=="vehicle"]
data_truck=ori_data[ori_data["class"]=="truck"]


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 速度
data_vehicle["s v x group"]=pd.cut(data_vehicle["s v x"],[0,15,20,25,30,35,40],labels=["0~15m/s","15~20m/s","20~25m/s","25~30m/s","30~35m/s","35~40m/s"])
data_vehicle_speed_group_1=data_vehicle[data_vehicle["s v x group"]=="0~15m/s"]
data_vehicle_speed_group_2=data_vehicle[data_vehicle["s v x group"]=="15~20m/s"]
data_vehicle_speed_group_3=data_vehicle[data_vehicle["s v x group"]=="20~25m/s"]
data_vehicle_speed_group_4=data_vehicle[data_vehicle["s v x group"]=="25~30m/s"]
data_vehicle_speed_group_5=data_vehicle[data_vehicle["s v x group"]=="30~35m/s"]
data_vehicle_speed_group_6=data_vehicle[data_vehicle["s v x group"]=="35~40m/s"]
gg_data_vehicle_speed_group_1 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_1["LCD"], data_vehicle_speed_group_1["complete"], label='0~15m/s(vehicle)')
gg_data_vehicle_speed_group_2 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_2["LCD"], data_vehicle_speed_group_2["complete"], label='15~20m/s(vehicle)')
gg_data_vehicle_speed_group_3 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_3["LCD"], data_vehicle_speed_group_3["complete"], label='20~25m/s(vehicle)')
gg_data_vehicle_speed_group_4 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_4["LCD"], data_vehicle_speed_group_4["complete"], label='25~30m/s(vehicle)')
gg_data_vehicle_speed_group_5 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_5["LCD"], data_vehicle_speed_group_5["complete"], label='30~35m/s(vehicle)')
gg_data_vehicle_speed_group_6 = GeneralizedGammaFitter().fit(data_vehicle_speed_group_6["LCD"], data_vehicle_speed_group_6["complete"], label='35~40m/s(vehicle)')
data_truck["s v x group"]=pd.cut(data_truck["s v x"],[0,15,20,25,30,35,40],labels=["0~15m/s","15~20m/s","20~25m/s","25~30m/s","30~35m/s","35~40m/s"])
data_truck_speed_group_1=data_truck[data_truck["s v x group"]=="0~15m/s"]
data_truck_speed_group_2=data_truck[data_truck["s v x group"]=="15~20m/s"]
data_truck_speed_group_3=data_truck[data_truck["s v x group"]=="20~25m/s"]
data_truck_speed_group_4=data_truck[data_truck["s v x group"]=="25~30m/s"]
data_truck_speed_group_5=data_truck[data_truck["s v x group"]=="30~35m/s"]
data_truck_speed_group_6=data_truck[data_truck["s v x group"]=="35~40m/s"]
gg_data_truck_speed_group_1 = GeneralizedGammaFitter().fit(data_truck_speed_group_1["LCD"], data_truck_speed_group_1["complete"], label='0~15m/s(truck)')
gg_data_truck_speed_group_2 = GeneralizedGammaFitter().fit(data_truck_speed_group_2["LCD"], data_truck_speed_group_2["complete"], label='15~20m/s(truck)')
gg_data_truck_speed_group_3 = GeneralizedGammaFitter().fit(data_truck_speed_group_3["LCD"], data_truck_speed_group_3["complete"], label='20~25m/s(truck)')
gg_data_truck_speed_group_4 = GeneralizedGammaFitter().fit(data_truck_speed_group_4["LCD"], data_truck_speed_group_4["complete"], label='25~30m/s(truck)')
gg_data_truck_speed_group_5 = GeneralizedGammaFitter().fit(data_truck_speed_group_5["LCD"], data_truck_speed_group_5["complete"], label='30~35m/s(truck)')
gg_data_truck_speed_group_6 = GeneralizedGammaFitter().fit(data_truck_speed_group_6["LCD"], data_truck_speed_group_6["complete"], label='35~40m/s(truck)')
print("小汽车速度分组")
print(gg_data_vehicle_speed_group_2.median_survival_time_)
print(gg_data_vehicle_speed_group_3.median_survival_time_)
print(gg_data_vehicle_speed_group_4.median_survival_time_)
print(gg_data_vehicle_speed_group_5.median_survival_time_)
print("卡车速度分组")
print(gg_data_truck_speed_group_2.median_survival_time_)
print(gg_data_truck_speed_group_3.median_survival_time_)
print(gg_data_truck_speed_group_4.median_survival_time_)
print(gg_data_truck_speed_group_5.median_survival_time_)








# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# thw
data_vehicle["dhw group"]=pd.cut(data_vehicle["dhw"],[0,50,100,150,200],labels=["0~50m","50~100m","100~150m","150~200m"])
data_vehicle_dhw_group_1=data_vehicle[data_vehicle["dhw group"]=="0~50m"]
data_vehicle_dhw_group_2=data_vehicle[data_vehicle["dhw group"]=="50~100m"]
data_vehicle_dhw_group_3=data_vehicle[data_vehicle["dhw group"]=="100~150m"]
data_vehicle_dhw_group_4=data_vehicle[data_vehicle["dhw group"]=="150~200m"]
gg_data_vehicle_dhw_group_1 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_1["LCD"], data_vehicle_dhw_group_1["complete"], label='0~50m(vehicle)')
gg_data_vehicle_dhw_group_2 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_2["LCD"], data_vehicle_dhw_group_2["complete"], label='50~100m(vehicle)')
gg_data_vehicle_dhw_group_3 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_3["LCD"], data_vehicle_dhw_group_3["complete"], label='100~150m(vehicle)')
gg_data_vehicle_dhw_group_4 = GeneralizedGammaFitter().fit(data_vehicle_dhw_group_4["LCD"], data_vehicle_dhw_group_4["complete"], label='150~200m(vehicle)')
data_truck["dhw group"]=pd.cut(data_truck["dhw"],[0,50,100,150,200],labels=["0~50m","50~100m","100~150m","150~200m"])
data_truck_dhw_group_1=data_truck[data_truck["dhw group"]=="0~50m"]
data_truck_dhw_group_2=data_truck[data_truck["dhw group"]=="50~100m"]
data_truck_dhw_group_3=data_truck[data_truck["dhw group"]=="100~150m"]
data_truck_dhw_group_4=data_truck[data_truck["dhw group"]=="150~200m"]
gg_data_truck_dhw_group_1 = GeneralizedGammaFitter().fit(data_truck_dhw_group_1["LCD"], data_truck_dhw_group_1["complete"], label='0~50m(truck)')
gg_data_truck_dhw_group_2 = GeneralizedGammaFitter().fit(data_truck_dhw_group_2["LCD"], data_truck_dhw_group_2["complete"], label='50~100m(truck)')
gg_data_truck_dhw_group_3 = GeneralizedGammaFitter().fit(data_truck_dhw_group_3["LCD"], data_truck_dhw_group_3["complete"], label='100~150m(truck)')
gg_data_truck_dhw_group_4 = GeneralizedGammaFitter().fit(data_truck_dhw_group_4["LCD"], data_truck_dhw_group_4["complete"], label='150~200m(truck)')
print("小汽车dhw分组")
print(gg_data_vehicle_dhw_group_1.median_survival_time_)
print(gg_data_vehicle_dhw_group_2.median_survival_time_)
print(gg_data_vehicle_dhw_group_3.median_survival_time_)
print(gg_data_vehicle_dhw_group_4.median_survival_time_)
print("卡车dhw分组")
print(gg_data_truck_dhw_group_1.median_survival_time_)
print(gg_data_truck_dhw_group_2.median_survival_time_)
print(gg_data_truck_dhw_group_3.median_survival_time_)
print(gg_data_truck_dhw_group_4.median_survival_time_)


# ————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# thw
data_vehicle["thw group"]=pd.cut(data_vehicle["thw"],[0,1,2,3,4],labels=["0~1s","1~2s","2~3s","3~4s"])
data_vehicle_thw_group_1=data_vehicle[data_vehicle["thw group"]=="0~1s"]
data_vehicle_thw_group_2=data_vehicle[data_vehicle["thw group"]=="1~2s"]
data_vehicle_thw_group_3=data_vehicle[data_vehicle["thw group"]=="2~3s"]
data_vehicle_thw_group_4=data_vehicle[data_vehicle["thw group"]=="3~4s"]
gg_data_vehicle_thw_group_1 = GeneralizedGammaFitter().fit(data_vehicle_thw_group_1["LCD"], data_vehicle_thw_group_1["complete"], label='0~1s(vehicle)')
gg_data_vehicle_thw_group_2 = GeneralizedGammaFitter().fit(data_vehicle_thw_group_2["LCD"], data_vehicle_thw_group_2["complete"], label='1~2s(vehicle)')
gg_data_vehicle_thw_group_3 = GeneralizedGammaFitter().fit(data_vehicle_thw_group_3["LCD"], data_vehicle_thw_group_3["complete"], label='2~3s(vehicle)')
gg_data_vehicle_thw_group_4 = GeneralizedGammaFitter().fit(data_vehicle_thw_group_4["LCD"], data_vehicle_thw_group_4["complete"], label='3~4s(vehicle)')
data_truck["thw group"]=pd.cut(data_truck["thw"],[0,1,2,3,4],labels=["0~1s","1~2s","2~3s","3~4s"])
data_truck_thw_group_1=data_truck[data_truck["thw group"]=="0~1s"]
data_truck_thw_group_2=data_truck[data_truck["thw group"]=="1~2s"]
data_truck_thw_group_3=data_truck[data_truck["thw group"]=="2~3s"]
data_truck_thw_group_4=data_truck[data_truck["thw group"]=="3~4s"]
gg_data_truck_thw_group_1 = GeneralizedGammaFitter().fit(data_truck_thw_group_1["LCD"], data_truck_thw_group_1["complete"], label='0~1s(truck)')
gg_data_truck_thw_group_2 = GeneralizedGammaFitter().fit(data_truck_thw_group_2["LCD"], data_truck_thw_group_2["complete"], label='1~2s(truck)')
gg_data_truck_thw_group_3 = GeneralizedGammaFitter().fit(data_truck_thw_group_3["LCD"], data_truck_thw_group_3["complete"], label='2~3s(truck)')
gg_data_truck_thw_group_4 = GeneralizedGammaFitter().fit(data_truck_thw_group_4["LCD"], data_truck_thw_group_4["complete"], label='3~4s(truck)')
print("小汽车thw分组")
print(gg_data_vehicle_thw_group_1.median_survival_time_)
print(gg_data_vehicle_thw_group_2.median_survival_time_)
print(gg_data_vehicle_thw_group_3.median_survival_time_)
print(gg_data_vehicle_thw_group_4.median_survival_time_)
print("卡车thw分组")
print(gg_data_truck_thw_group_1.median_survival_time_)
print(gg_data_truck_thw_group_2.median_survival_time_)
print(gg_data_truck_thw_group_3.median_survival_time_)
print(gg_data_truck_thw_group_4.median_survival_time_)









