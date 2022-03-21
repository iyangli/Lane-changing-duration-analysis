import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import numpy as np
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
from lifelines.plotting import add_at_risk_counts
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter,GeneralizedGammaRegressionFitter
from lifelines import CoxPHFitter
import pandas as pd
from lifelines import CoxPHFitter
font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 18,
}

ori_data=pd.read_excel("D:\\Paper\\2021,research\\2021,LCD\\data\\Analysis\\20210524,total data(together).xlsx")

ori_data["relative speed"]=ori_data["s v x"]-ori_data["p v x"]
data_vehicle=ori_data[ori_data["class"]=="vehicle"][["s v x","dhw","thw","LCD","complete"]]
data_truck=ori_data[ori_data["class"]=="truck"][["s v x","dhw","thw","LCD","complete"]]

data_vehicle_new=pd.DataFrame()
data_vehicle_new["distance-headway"]=data_vehicle["dhw"]
data_vehicle_new["time-headway"]=data_vehicle["thw"]
data_vehicle_new["speed"]=data_vehicle["s v x"]
data_vehicle_new["LCD"]=data_vehicle["LCD"]
data_vehicle_new["complete"]=data_vehicle["complete"]

data_truck_new=pd.DataFrame()
data_truck_new["distance-headway"]=data_truck["dhw"]
data_truck_new["time-headway"]=data_truck["thw"]
data_truck_new["speed"]=data_truck["s v x"]
data_truck_new["LCD"]=data_truck["LCD"]
data_truck_new["complete"]=data_truck["complete"]

lnf_vehicle = LogLogisticAFTFitter().fit(data_vehicle_new, 'LCD', 'complete')
lnf_truck  = LogLogisticAFTFitter().fit(data_truck_new, 'LCD', 'complete')



plt.figure(figsize=(14,12))
ax1 = plt.subplot(221)
lnf_vehicle.plot_partial_effects_on_outcome('speed', range(10, 35, 5), cmap='Accent_r',plot_baseline=True)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(a) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("speed of passenger cars",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

ax2 = plt.subplot(222)
lnf_vehicle.plot_partial_effects_on_outcome('time-headway', range(0, 4, 1), cmap='Accent_r',plot_baseline=True)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(b) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("time-headway of passenger cars",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

ax3 = plt.subplot(223)
lnf_vehicle.plot_partial_effects_on_outcome('distance-headway', range(0, 200, 50), cmap='Accent_r',plot_baseline=True)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(c) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("distance-headway of passenger cars",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

ax4 = plt.subplot(224)
lnf_truck.plot_partial_effects_on_outcome('speed', range(10, 35, 5), cmap='Accent_r',plot_baseline=True)
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.xticks([0,2,4,6,8,10,12])
plt.xlabel("(d) Timeline(s)",font1)
plt.ylabel("Probability",font1)
plt.title("speed of heavy vehicles",font1)
plt.grid(linestyle='--')
sns.set_context("notebook")

plt.show()
plt.savefig('D:\\Paper\\2021,research\\2021,LCD\\results\\visulization\\20210625,AFT partial results.png',bbox_inches='tight',dpi=800)





