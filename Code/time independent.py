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

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)



ori_data=pd.read_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20201005,resampled data(0.1s).xlsx")
ori_data["status"]=1
base_table=ori_data[ori_data["interval"]==0]


ctv_table=base_table[['s v x','s a x','p a x','tp a x','f a x','tf a x',"status",'initial disdiff tpf',
                            "disdiff p", 'disdiff tp',"disdiff f","disdiff tf",
                            "spediff p x","spediff tp x","spediff f x","spediff tf x","LCD"]]

# ctv_table.to_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20201007,time-independent data(AFT model).xlsx")
# PH假设模型
cph=CoxPHFitter()
cph.fit(ctv_table, duration_col="LCD", event_col="status")
cph.print_summary()

# cph.summary.to_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\AFT\\conf lifelines.xlsx")

cph._predicted_partial_hazards_
from lifelines.statistics import proportional_hazard_test
results = proportional_hazard_test(cph, ctv_table, time_transform='rank')

cph.plot_covariate_groups()

a=cph.check_assumptions(ctv_table, show_plots=True)

cph.predict_survival_function(ctv_table)
cph.predict_median(ctv_table)
cph.predict_partial_hazard(ctv_table)

cph.plot_covariate_groups(covariates='disdiff p', values=[50, 100, 150, 200], cmap='coolwarm')
plt.show()
cph.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()

# AFT类模型

llf = LogLogisticAFTFitter().fit(ctv_table,  duration_col='LCD', event_col='status')
lnf = LogNormalAFTFitter().fit(ctv_table,  duration_col='LCD', event_col='status')
wf = WeibullAFTFitter().fit(ctv_table,  duration_col='LCD', event_col='status')

llf.summary.to_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\AFT\\llf lifelines.xlsx")
lnf.summary.to_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\AFT\\lnf lifelines.xlsx")
wf.summary.to_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\AFT\\wf lifelines.xlsx")

wf.plot_covariate_groups(covariates='disdiff p', values=[0,25,50,75,100,125,150], cmap='coolwarm')
plt.show()

llf.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()

lnf.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()

wf.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()

llf.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()

lnf.plot_covariate_groups(covariates='spediff tf x', values=[-15,-10,-5,0,5,10,15], cmap='coolwarm')
plt.show()


# ggf = GeneralizedGammaRegressionFitter().fit(ctv_table,  duration_col='LCD', event_col='status')

# s.v.x+s.a.x+p.a.x+tp.a.x+ f.a.x+ tf.a.x+ initial.disdiff.tpf+ disdiff.p+ disdiff.tp+ disdiff.f+disdiff.tf+ spediff.p.x+ spediff.tp.x+ spediff.f.x+spediff.tf.x
