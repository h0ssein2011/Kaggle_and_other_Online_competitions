
import pandas as pd
import numpy as np

array = np.random.random((36, 36))
array1 = np.random.random((36, 10))
df = pd.DataFrame(array)
df2 = pd.DataFrame(array1)
print("Put breakpoint here")
df[0][0] = 1
print("The End")