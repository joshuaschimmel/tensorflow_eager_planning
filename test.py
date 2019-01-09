import pandas as pd
import numpy as np

df = pd.read_csv("pendulum_data.csv")
print(df.head().values.std(axis=0))
