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
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows',None)



ori_data=pd.read_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20201005,resampled data(0.1s).xlsx")
ori_data["status"]=1


# Time-dependent————————————————————————————————————————————————————————————————————————————————
# base_table_1，主表

base_table=ori_data[ori_data["interval"]==0]


# a=base_table_1[["cur_id","spediff_f","spediff_tf","spediff_tr","dis_f", "dis_tf","dis_tr","dis_tr-tf", 's_a', 's_v','f_a', 'f_v', 'tf_a', 'tf_v','tr_a', 'tr_v','lc_t','label',"status"]]
# a.to_csv("C:\\Users\\cc\\Desktop\\no-time_dependent_more.csv")

# base_table_1=base_table[["lc id","LCD","status","initial disdiff tpf","initial disdiff p","initial spe"]].sort_values(by="lc id")
base_table_1=base_table[["lc id","LCD","status"]].sort_values(by="lc id")


from lifelines.utils import to_long_format
base_df_main = to_long_format(base_table_1, duration_col="LCD")
# base_table_2,副表
from lifelines.utils import add_covariate_to_timeline

base_df_secondary=ori_data[["lc id","interval",'s v x','s a x','p a x','tp a x','f a x','tf a x',
                            "disdiff p", 'disdiff tp',"disdiff f","disdiff tf",
                            "spediff p x","spediff tp x","spediff f x","spediff tf x"]]

df = add_covariate_to_timeline(base_df_main, base_df_secondary, duration_col="interval", id_col="lc id", event_col="status")
df.to_excel("D:\\Paper\\2020,research\\2020年，LCD（中文）\\results\\2020203,df.xlsx")

# r_survival=df
# r_survival["subject"]=r_survival["cur_id"]
# r_survival.to_csv("C:\\Users\\cc\\Desktop\\time_dependent_more.csv")

from lifelines import CoxTimeVaryingFitter
from lifelines.utils import restricted_mean_survival_time

ctv = CoxTimeVaryingFitter()
ctv.fit(df, id_col="lc id", event_col="status",start_col="start", stop_col="stop", show_progress=True)

from lifelines.statistics import proportional_hazard_test
results = proportional_hazard_test(ctv, df, time_transform='rank')
ctv.check_assumptions(df, show_plots=True)


ctv.print_summary()
ctv.summary.to_excel("D:\\Paper\\2020,research\\2020年，LCD（中文）\\results\\time-dependent.xlsx")
ctv.check_assumptions(df, p_value_threshold=0.05, show_plots=True)

ctv.baseline_survival_.plot()
ctv.baseline_cumulative_hazard_.plot()

plt.show()

ctv.plot_covariate_groups("spediff_f",[-5,-3,-1,1,3,5])

print(restricted_mean_survival_time(ctv))
# Time-dependent————————————————————————————————————————————————————————————————————————————————

# ctv.compute_residuals(df,"schoenfeld")





