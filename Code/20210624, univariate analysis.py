import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import numpy as np
from lifelines import WeibullAFTFitter
import math
import pandas as pd
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)
import seaborn as sns
from scipy.stats import norm #使用直方图和最大似然高斯分布拟合绘制分布
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from numpy import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
import seaborn as sns
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 25)
import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter,NelsonAalenFitter
import numpy as np
import math
from lifelines import WeibullFitter
from lifelines import (WeibullFitter, ExponentialFitter,
LogNormalFitter, LogLogisticFitter, NelsonAalenFitter,
PiecewiseExponentialFitter, GeneralizedGammaFitter, SplineFitter)
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
from lifelines.utils import median_survival_times
from lifelines.utils import find_best_parametric_model
from lifelines.plotting import qq_plot
from lifelines.utils import restricted_mean_survival_time

font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

ori_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\LC_Extraction\\20210524,total data(together).xlsx")
ori_data["relative speed"]=ori_data["s v x"]-ori_data["p v x"]

data_vehicle=ori_data[ori_data["class"]=="vehicle"]
data_truck=ori_data[ori_data["class"]=="truck"]

T_vehicle=data_vehicle["LCD"]
E_vehicle=data_vehicle["complete"]

T_truck=data_truck["LCD"]
E_truck=data_truck["complete"]

best_model_vehicle, best_aic__vehicle = find_best_parametric_model(T_vehicle, E_vehicle, scoring_method="BIC",show_progress=True)
a_vehicle=find_best_parametric_model(T_vehicle, E_vehicle, scoring_method="BIC")
wbf_vehicle = WeibullFitter().fit(T_vehicle, E_vehicle, label='Weibull')
exf_vehicle = ExponentialFitter().fit(T_vehicle, E_vehicle, label='Exponential')
lnf_vehicle = LogNormalFitter().fit(T_vehicle, E_vehicle, label='LogNormal')
llf_vehicle = LogLogisticFitter().fit(T_vehicle, E_vehicle, label='LogLogistic')
gg_vehicle = GeneralizedGammaFitter().fit(T_vehicle, E_vehicle, label='GeneralizedGamma')

best_model_truck, best_aic__truck = find_best_parametric_model(T_truck, E_truck, scoring_method="BIC",show_progress=True)
a_truck=find_best_parametric_model(T_truck, E_truck, scoring_method="BIC")
wbf_truck = WeibullFitter().fit(T_truck, E_truck, label='Weibull')
exf_truck = ExponentialFitter().fit(T_truck, E_truck, label='Exponential')
lnf_truck = LogNormalFitter().fit(T_truck, E_truck, label='LogNormal')
llf_truck = LogLogisticFitter().fit(T_truck, E_truck, label='LogLogistic')
gg_truck = GeneralizedGammaFitter().fit(T_truck, E_truck, label='GeneralizedGamma')

gg_vehicle_1 = GeneralizedGammaFitter().fit(T_vehicle, E_vehicle, label='passenger cars')
gg_truck_1 = GeneralizedGammaFitter().fit(T_truck, E_truck, label='heavy vehicles')


rmst_12=[]
time=[]
for i in range(0,120):
    rmst_group_1=restricted_mean_survival_time(gg_vehicle, t=i/10)
    rmst_group_2=restricted_mean_survival_time(gg_truck, t=i/10)
    rmst_12.append(rmst_group_1-rmst_group_2)
    time.append(i/10)

rmst_group=pd.DataFrame({"time":time,"Group1-Group2":rmst_12})


plt.figure(figsize=(12,10))
ax1 = plt.subplot(221)
wbf_vehicle.plot_survival_function(ci_show=False)
exf_vehicle.plot_survival_function(ci_show=False)
lnf_vehicle.plot_survival_function(ci_show=False)
llf_vehicle.plot_survival_function(ci_show=False)
gg_vehicle.plot_survival_function(ci_show=False)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(a) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("survival function(passenger cars)",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

ax2 = plt.subplot(222)
wbf_truck.plot_survival_function(ci_show=False)
exf_truck.plot_survival_function(ci_show=False)
lnf_truck.plot_survival_function(ci_show=False)
llf_truck.plot_survival_function(ci_show=False)
gg_truck.plot_survival_function(ci_show=False)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(b) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("survival function(heavy vehicles)",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

ax3 = plt.subplot(223)
gg_vehicle_1.plot_survival_function(ci_show=False,c="blue")
gg_truck_1.plot_survival_function(ci_show=False,c="red")
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(c) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("survival function(comparison)",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.axvline(gg_vehicle.median_survival_time_ , color='blue' , linestyle='--')
average_1=round(gg_vehicle.median_survival_time_,2)
plt.text(6.6, 0.53, 'MST='+str(average_1)+"s",color="blue",fontdict=font1)
plt.axvline(gg_truck.median_survival_time_ , color='red' , linestyle='--')
average_2=round(gg_truck.median_survival_time_,2)
plt.text(6.6, 0.65, 'MST='+str(average_2)+"s",color="red",fontdict=font1)
plt.axhline(0.5, color='black' , linestyle='--')

ax4 = plt.subplot(224)
from matplotlib.ticker import FuncFormatter
def to_percent(temp, position):
    return '%1.00f'%(100*temp) + '%'

at_risk_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\Analysis\\20210525,at risk counts.xlsx",sheet_name="Sheet2")
plt.plot(at_risk_data["time"],at_risk_data["difference"],color="blue",label="heavy vehicles minus passenger cars")
plt.xticks([0,2,4,6,8,10,12])
plt.yticks(np.arange(-0.05,0.3,0.05))
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.xlabel("(d)Timeline(s)",font1)
plt.ylabel("Percentage",font1)
plt.title("At risk percentage difference",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.legend()




plt.tight_layout()

plt.savefig('D:\\Paper\\2021,research\\2021,LCD\\results\\visulization\\20210624,univariate comparison.png',bbox_inches='tight',dpi=800)




