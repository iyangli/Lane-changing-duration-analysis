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
from lifelines import GeneralizedGammaFitter
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


ori_data=pd.read_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20200922,first head of the LC data(threhold=1.5).xlsx",sheet_name="Sheet1")


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 速度划分
ori_data["initial s v x group"]=pd.cut(ori_data["abs(s v x)"],[0,10,20,30,40],labels=["0~10m/s","10~20m/s","20~30m/s","30~40m/s"])

group_1_s_v_group=ori_data[ori_data["initial s v x group"]=="0~10m/s"]
group_2_s_v_group=ori_data[ori_data["initial s v x group"]=="10~20m/s"]
group_3_s_v_group=ori_data[ori_data["initial s v x group"]=="20~30m/s"]
group_4_s_v_group=ori_data[ori_data["initial s v x group"]=="30~40m/s"]


kmf_1_s_v_group = GeneralizedGammaFitter().fit(group_1_s_v_group["LCD"], group_1_s_v_group["complete"], label='0~10m/s')
kmf_2_s_v_group = GeneralizedGammaFitter().fit(group_2_s_v_group["LCD"], group_2_s_v_group["complete"], label='10~20m/s')
kmf_3_s_v_group = GeneralizedGammaFitter().fit(group_3_s_v_group["LCD"], group_3_s_v_group["complete"], label='20~30m/s')
kmf_4_s_v_group = GeneralizedGammaFitter().fit(group_4_s_v_group["LCD"], group_4_s_v_group["complete"], label='30~40m/s')

time=[]
rmst_s_v_1=[]
rmst_s_v_2=[]
rmst_s_v_3=[]
rmst_s_v_4=[]

for i in range(0,140):
    rmst_group_1=restricted_mean_survival_time(kmf_1_s_v_group, t=i/10)
    rmst_group_2=restricted_mean_survival_time(kmf_2_s_v_group, t=i/10)
    rmst_group_3=restricted_mean_survival_time(kmf_3_s_v_group, t=i/10)
    rmst_group_4=restricted_mean_survival_time(kmf_4_s_v_group, t=i/10)
    rmst_s_v_1.append(rmst_group_1)
    rmst_s_v_2.append(rmst_group_2)
    rmst_s_v_3.append(rmst_group_3)
    rmst_s_v_4.append(rmst_group_4)
    time.append(i/10)

rmst_group_s_v=pd.DataFrame({"time":time,"rmst_0~10m/s":rmst_s_v_1,"rmst_10~20m/s":rmst_s_v_2,"rmst_20~30m/s":rmst_s_v_3,"rmst_30~40m/s":rmst_s_v_4})


# ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# 距离划分
ori_data["initial disdiff p group"]=pd.cut(ori_data["disdiff p"],[0,25,50,75,100,180],labels=["0~25m","25~50m","50~75m","75~100m",">100m"])

group_1_disdiff_group=ori_data[ori_data["initial disdiff p group"]=="0~25m"]
group_2_disdiff_group=ori_data[ori_data["initial disdiff p group"]=="25~50m"]
group_3_disdiff_group=ori_data[ori_data["initial disdiff p group"]=="50~75m"]
group_4_disdiff_group=ori_data[ori_data["initial disdiff p group"]=="75~100m"]
group_5_disdiff_group=ori_data[ori_data["initial disdiff p group"]==">100m"]


kmf_1_disdiff_group = GeneralizedGammaFitter().fit(group_1_disdiff_group["LCD"], group_1_disdiff_group["complete"], label='0~25m')
kmf_2_disdiff_group = GeneralizedGammaFitter().fit(group_2_disdiff_group["LCD"], group_2_disdiff_group["complete"], label='25~50m')
kmf_3_disdiff_group = GeneralizedGammaFitter().fit(group_3_disdiff_group["LCD"], group_3_disdiff_group["complete"], label='50~75m')
kmf_4_disdiff_group = GeneralizedGammaFitter().fit(group_4_disdiff_group["LCD"], group_4_disdiff_group["complete"], label='75~100m')
kmf_5_disdiff_group = GeneralizedGammaFitter().fit(group_5_disdiff_group["LCD"], group_5_disdiff_group["complete"], label='>100m')

time=[]
rmst_disdiff_1=[]
rmst_disdiff_2=[]
rmst_disdiff_3=[]
rmst_disdiff_4=[]
rmst_disdiff_5=[]

for i in range(0,140):
    rmst_group_1=restricted_mean_survival_time(kmf_1_disdiff_group, t=i/10)
    rmst_group_2=restricted_mean_survival_time(kmf_2_disdiff_group, t=i/10)
    rmst_group_3=restricted_mean_survival_time(kmf_3_disdiff_group, t=i/10)
    rmst_group_4=restricted_mean_survival_time(kmf_4_disdiff_group, t=i/10)
    rmst_group_5=restricted_mean_survival_time(kmf_5_disdiff_group, t=i/10)
    rmst_disdiff_1.append(rmst_group_1)
    rmst_disdiff_2.append(rmst_group_2)
    rmst_disdiff_3.append(rmst_group_3)
    rmst_disdiff_4.append(rmst_group_4)
    rmst_disdiff_5.append(rmst_group_5)

    time.append(i/10)

rmst_group_disdiff=pd.DataFrame({"time":time,"0~25m":rmst_disdiff_1,"25~50m":rmst_disdiff_2,"50~75m":rmst_disdiff_3,"75~100m":rmst_disdiff_4,">100m":rmst_disdiff_5})



plt.figure(figsize=(12,10))

ax1 = plt.subplot(221)
kmf_1_s_v_group.plot_survival_function(ci_show=False)
kmf_2_s_v_group.plot_survival_function(ci_show=False)
kmf_3_s_v_group.plot_survival_function(ci_show=False)
kmf_4_s_v_group.plot_survival_function(ci_show=False)
# plt.annotate(xy=(kmf_1_s_v_group.median_survival_time_, 0.5), )  # 添加注释
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12,14])
plt.xlabel("a) Timeline(s)")
plt.ylabel("Probability")
plt.title("different initial speed")
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.legend()

ax2 = plt.subplot(222)
plt.plot(rmst_group_s_v["time"],rmst_group_s_v["rmst_0~10m/s"],label="0~10m/s")
plt.plot(rmst_group_s_v["time"],rmst_group_s_v["rmst_10~20m/s"],label="10~20m/s")
plt.plot(rmst_group_s_v["time"],rmst_group_s_v["rmst_20~30m/s"],label="20~30m/s")
plt.plot(rmst_group_s_v["time"],rmst_group_s_v["rmst_30~40m/s"],label="30~40m/s")
plt.yticks([0,1,2,3,4,5,6,7,8])
plt.xticks([0,2,4,6,8,10,12,14])
plt.xlabel("b)Timeline(s)")
plt.ylabel("RMST")
plt.title("different initial speed")
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.legend()

ax3 = plt.subplot(223)

kmf_1_disdiff_group.plot_survival_function(ci_show=False)
kmf_2_disdiff_group.plot_survival_function(ci_show=False)
kmf_3_disdiff_group.plot_survival_function(ci_show=False)
kmf_4_disdiff_group.plot_survival_function(ci_show=False)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12,14])
plt.xlabel("c) Timeline(s)")
plt.ylabel("Probability")
plt.title("different initial distance with preceding vehicle")
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.legend()

ax4 = plt.subplot(224)
plt.plot(rmst_group_disdiff["time"],rmst_group_disdiff["0~25m"],label="0~25m")
plt.plot(rmst_group_disdiff["time"],rmst_group_disdiff["25~50m"],label="25~50m")
plt.plot(rmst_group_disdiff["time"],rmst_group_disdiff["50~75m"],label="50~75m")
plt.plot(rmst_group_disdiff["time"],rmst_group_disdiff["75~100m"],label="75~100m")
plt.yticks([0,1,2,3,4,5,6])
plt.xticks([0,2,4,6,8,10,12,14])
plt.xlabel("d) Timeline(s)")
plt.ylabel("RMST")
plt.title("different initial distance with preceding vehicle")
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.legend()
plt.tight_layout()

plt.savefig("D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\survival rmse difference(initial state)-2.png",bbox_inches='tight',dpi=600)

len(group_1_s_v_group)
len(group_2_s_v_group)
len(group_3_s_v_group)
len(group_4_s_v_group)

len(group_1_disdiff_group)
len(group_2_disdiff_group)
len(group_3_disdiff_group)
len(group_4_disdiff_group)

print(median_survival_times(kmf_1_s_v_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_2_s_v_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_3_s_v_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_4_s_v_group.confidence_interval_survival_function_))

print(median_survival_times(kmf_1_disdiff_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_2_disdiff_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_3_disdiff_group.confidence_interval_survival_function_))
print(median_survival_times(kmf_4_disdiff_group.confidence_interval_survival_function_))

