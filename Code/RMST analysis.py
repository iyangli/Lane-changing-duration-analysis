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
from lifelines.plotting import rmst_plot
from lifelines.utils import restricted_mean_survival_time
font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

ori_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\LC_Extraction\\20210524,total data(together).xlsx")

data_vehicle=ori_data[ori_data["class"]=="vehicle"]
data_truck=ori_data[ori_data["class"]=="truck"]

T_vehicle=data_vehicle["LCD"]
E_vehicle=data_vehicle["complete"]

T_truck=data_truck["LCD"]
E_truck=data_truck["complete"]

time_limit = 50
gg_vehicle = GeneralizedGammaFitter().fit(T_vehicle, E_vehicle, label='Vehicle')
gg_truck = GeneralizedGammaFitter().fit(T_truck, E_truck, label='Truck')


rmst_12=[]
time=[]
for i in range(0,120):
    rmst_group_1=restricted_mean_survival_time(gg_vehicle, t=i/10)
    rmst_group_2=restricted_mean_survival_time(gg_truck, t=i/10)
    rmst_12.append(rmst_group_1-rmst_group_2)
    time.append(i/10)

rmst_group=pd.DataFrame({"time":time,"Group1-Group2":rmst_12})



plt.figure(figsize=(10,5))
ax1 = plt.subplot(121)
gg_vehicle.plot_survival_function(ci_show=False,c="blue")
gg_truck.plot_survival_function(ci_show=False,c="red")
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(a) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("survival function(comparison)",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.axvline(gg_vehicle.median_survival_time_ , color='blue' , linestyle='--')
average_1=round(gg_vehicle.median_survival_time_,2)
plt.text(6.6, 0.53, 'MST='+str(average_1)+"s",color="blue")
plt.axvline(gg_truck.median_survival_time_ , color='red' , linestyle='--')
average_2=round(gg_truck.median_survival_time_,2)
plt.text(6.6, 0.65, 'MST='+str(average_2)+"s",color="red")
plt.axhline(0.5, color='black' , linestyle='--')

ax2 = plt.subplot(122)
plt.plot(rmst_group["time"],rmst_group["Group1-Group2"],color="blue")
plt.xticks([0,2,4,6,8,10,12])
plt.yticks([-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2])
plt.xlabel("(b)Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("RMST function",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")
plt.axvline(3.5, color='black' , linestyle='--')
plt.axvline(8.5, color='black' , linestyle='--')
# plt.axhline(0.1, color='black' , linestyle='--')
# plt.axhline(-0.6, color='black' , linestyle='--')

# plt.show()


plt.savefig('D:\\Paper\\2021,research\\2021,LCD\\results\\visulization\\univariate comparison(trucks and vehicles)-2.png',bbox_inches='tight',dpi=600)



