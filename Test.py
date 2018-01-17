import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.rand(140, 4), columns=['A', 'B', 'C', 'D'])
df.A = df.A* 10
df.B = df.B*100
df.C = df.C*1000
df.D = df.D* 100000

df['models'] = pd.Series(np.repeat(['model1','model2', 'model3', 'model4', 'model5', 'model6', 'model7'], 20))

df
fig, ax_new = plt.subplots(2,2, sharey=False)

bp = df.boxplot(by="models",ax=ax_new,figsize=(6,8))

#[ax_tmp.set_xlabel('') for ax_tmp in ax_new.reshape(-1)]

#[ax_tmp.set_ylim(-2, 2) for ax_tmp in ax_new[1]]

#fig.suptitle('New title here')

plt.show()
