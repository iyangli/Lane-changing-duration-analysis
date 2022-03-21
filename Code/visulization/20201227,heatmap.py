import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter
from lifelines import WeibullAFTFitter
import numpy as np
import math

# 开始聚类——————————————————————————————————————————————————————————————————————————————————————————
from sklearn import metrics
from sklearn.cluster import AffinityPropagation,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.stats import kstest

ori_data=pd.read_excel("D:\\Paper\\2020,research\\2020,LCD\\dataset\\HighD dataset\\20200922,first head of the LC data(threhold=1.5).xlsx",sheet_name="20201227,Sheet3")



# spearman 以及皮尔逊系数分别为
import seaborn as sns
b=pd.DataFrame()
a=ori_data

b["LCD"]=a["LCD"]
b["Acc S"]=a["s a x"]
b["Spe S"]=a["s v x"]

b["Acc CP"]=a["p a x"]
b["Spe CP"]=a["p v x"]

b["Acc TP"]=a["tp a x"]
b["Spe TP"]=a["tp v x"]

b["Acc CF"]=a["f a x"]
b["Spe CF"]=a["f v x"]

b["Acc TF"]=a["tf a x"]
b["Spe TF"]=a["tf v x"]

b["Spediff S-CP"]=a["spediff p x"]
b["Spediff S-TP"]=a["spediff tp x"]
b["Spediff S-CF"]=a["spediff f x"]
b["Spediff S-TF"]=a["spediff tf x"]

b["Disdiff S-CP"]=a["disdiff p"]
b["Disdiff S-TP"]=a["disdiff tp"]
b["Disdiff S-CF"]=a["disdiff f"]
b["Disdiff S-TF"]=a["disdiff tf"]
b["Disdiff TP-TF"]=a["dis tpf"]


fontsize=12

font1 = {
    'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : fontsize,
}
corr_table=b.corr()
spearman_table=b.corr("spearman")


plt.figure(figsize=(10,5),dpi=200)
plt.figure(1)
ax1 = plt.subplot(121)
# sns.heatmap(spearman_table,cmap='rainbow',center=0.1,annot=True, annot_kws={'size':9,'weight':'bold', 'color':'blue'})
# sns.heatmap(spearman_table,cmap='rainbow',center=0.1,linewidths = 0.05,linecolor= 'red', mask=(spearman_table<0.3))
sns.heatmap(spearman_table,cmap='rainbow', cbar=False,center=0.1,linewidths = 1,linecolor= 'black')
# plt.set_title('matplotlib colormap')
plt.xticks([])
plt.xlabel("(a) without mask",font1)

ax2 = plt.subplot(122)
sns.heatmap(spearman_table,cmap='rainbow',center=0.1, mask=(spearman_table<0.2),yticklabels=False,linewidths = 1,linecolor= 'black')
plt.xticks([])
plt.xlabel("(b) mask < 0.2",font1)
# plt.suptitle("Spearman Heatmap")
plt.tight_layout()
plt.savefig("D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\20201227,heatmap_variable.png",bbox_inches='tight',dpi=200)
plt.close('all')

# plt.show()
#
spearman_table.to_excel("D:\\Paper\\2020,research\\2020,LCD\\results\\HighD result\\20201227,spearman_table.xlsx")
