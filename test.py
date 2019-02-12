import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import helper_functions as _hp


df = pd.read_csv("data/pendulum_data_dot_stretched.csv")

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# partition the data into a training and testing set
partition_index = int(0.1 * len(df.index))
train_df = df.iloc[partition_index:, :]
test_df = df.iloc[:partition_index, :]


# TODO normalize dot and action
# strip the data and labels from the training sets
features = train_df.loc[:, [
                               "s_0_cos(theta)",
                               "s_0_sin(theta)",
                               "s_0_theta_dot",
                               "action"
                           ]]
labels = train_df.loc[:, [
                                 "s_1_cos(theta)",
                                 "s_1_sin(theta)",
                                 "s_1_theta_dot"
                             ]]

print(np.max(labels.loc[:, ["s_1_theta_dot"]]))
print(np.min(labels.loc[:, ["s_1_theta_dot"]]))

feat_dot = features.loc[:, ["s_0_theta_dot"]]
lab_dot = labels.loc[:, ["s_1_theta_dot"]]
ax = lab_dot[:].plot(kind="kde", label="State 0")
feat_dot[:].plot(kind="kde", label="State 1", ax=ax)
plt.title("Distribution of theta_dot")
plt.show()

# normalizations
features.loc[:, "s_0_theta_dot"] = features.loc[:, "s_0_theta_dot"].apply(
    lambda x: _hp.min_max_norm(v=x,
                               v_min=-8, v_max=8,
                               n_min=-1, n_max=1
                               )
)
features.loc[:, "action"] = features.loc[:, "action"].apply(
    lambda x: _hp.min_max_norm(v=x,
                               v_min=-2, v_max=2,
                               n_min=-1, n_max=1
                               )
)

#features.plot()
#plt.show()