import pandas as pd


df = pd.read_csv("data/pendulum_data.csv")

# shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# partition the data into a training and testing set
partition_index = int(0.1 * len(df.index))
train_df = df.iloc[partition_index:, :]
test_df = df.iloc[:partition_index, :]

 # TODO normalize dot and action
# strip the data and labes from the training sets
features = train_df.loc[:, [
                               "s_0_cos(theta)",
                               "s_0_sin(theta)",
                               "s_0_theta_dot",
                               "action"
                           ]].values

labels = train_df.loc[:, [
                             "s_1_cos(theta)",
                             "s_1_sin(theta)",
                             "s_1_theta_dot"
                         ]].values

# do the same for the test data
test_features = test_df.loc[:, [
                                   "s_0_cos(theta)",
                                   "s_0_sin(theta)",
                                   "s_0_theta_dot",
                                   "action"
                               ]].values

test_labels = test_df.loc[:, ["s_1_cos(theta)",
                              "s_1_sin(theta)",
                              "s_1_theta_dot"
                              ]].values