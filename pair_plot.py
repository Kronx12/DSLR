from numpy.core.numeric import full
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from libs.dataset_reader import *
from libs.lib_stat import *

header, data, raw_header, raw = open_dataset("datasets/dataset_train.csv")

df = pd.DataFrame(data=data, index=list(range(len(data))), columns=header)
df2 = pd.DataFrame(data=raw, index=list(range(len(data))), columns=raw_header)
df = pd.concat([df2, df], axis=1)
print(df)
# print(df)
sns.set(style="ticks", color_codes=True)
sns.pairplot(df, hue="Hogwarts House")
plt.savefig("exports/export_pair_plot.png", dpi=100)
plt.show()