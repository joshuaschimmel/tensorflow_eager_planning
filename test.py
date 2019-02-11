import pandas as pd
import helper_functions as _hp


df = pd.read_csv("data/pendulum_data.csv")

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

features.plot()

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

features.plot()