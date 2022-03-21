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
# data_vehicle=ori_data[ori_data["class"]=="vehicle"][["s v x","relative speed","dhw","thw","LCD","complete"]]
# data_truck=ori_data[ori_data["class"]=="truck"][["s v x","relative speed","dhw","thw","LCD","complete"]]

data_vehicle=ori_data[ori_data["class"]=="vehicle"][["s v x","dhw","thw","LCD","complete"]]
data_truck=ori_data[ori_data["class"]=="truck"][["s v x","dhw","thw","LCD","complete"]]


llf_vehicle = LogLogisticAFTFitter().fit(data_vehicle, 'LCD', 'complete')
lnf_vehicle = LogNormalAFTFitter().fit(data_vehicle, 'LCD', 'complete')
wf_vehicle = WeibullAFTFitter().fit(data_vehicle, 'LCD', 'complete')

print(llf_vehicle.AIC_,lnf_vehicle.AIC_,wf_vehicle.AIC_)
print(llf_vehicle.BIC_,lnf_vehicle.BIC_,wf_vehicle.BIC_)
print(llf_vehicle.median_survival_time_,lnf_vehicle.median_survival_time_,wf_vehicle.median_survival_time_)
llf_vehicle.print_summary()
lnf_vehicle.print_summary()
wf_vehicle.print_summary()



llf_truck = LogLogisticAFTFitter().fit(data_truck, 'LCD', 'complete')
lnf_truck  = LogNormalAFTFitter().fit(data_truck, 'LCD', 'complete')
wf_truck  = WeibullAFTFitter().fit(data_truck, 'LCD', 'complete')

print(llf_truck.AIC_,lnf_truck.AIC_,wf_truck.AIC_)
print(llf_truck.BIC_,lnf_truck.BIC_,wf_truck.BIC_)
print(llf_truck.median_survival_time_,lnf_truck.median_survival_time_,wf_truck.median_survival_time_)


llf_truck.print_summary()
lnf_truck.print_summary()
wf_truck.print_summary()






