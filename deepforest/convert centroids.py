import pandas as pd

df = pd.read_csv('deepforest/data/OSBS_029.csv')

# Calculate centroids
df['x_center'] = round((df['xmin'] + df['xmax']) / 2)
df['y_center'] = round((df['ymin'] + df['ymax']) / 2)

# Drop original bounding box columns
df_centroids = df.drop(columns=["xmin", "ymin", "xmax", "ymax"])
df_centroids.to_csv("deepforest/data/OSBS_029_centroids.csv")
