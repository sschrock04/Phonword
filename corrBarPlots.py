import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("C:\\Users\\Stephen\\Downloads\\Phonword Correlation (100 sims) - grouped by binding.csv")
# Create DataFrame
#df = pd.DataFrame(data)
df["Orthographic Normalized Levenshtein Distance"] = df["Orthographic Normalized Levenshtein Distance"].abs()
df["Phonological Normalized Levenshtein Distance"] = df["Phonological Normalized Levenshtein Distance"].abs()
print(df)
df = df.drop([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50])
tr_indices = [0, 2, 6, 8, 12]
o_indices = [1, 3, 7, 9, 13]
tro_indices = [14, 16]
df_tr = df.iloc[tr_indices]
df_o = df.iloc[o_indices]
df_tro = df.iloc[tro_indices]
print(df)


# Plot settings
plt.figure(figsize=(16, 12))
'''
# Plot 1: Human-Rated Phonological Similarity
plt.subplot(3, 1, 1)
bars1 = plt.bar(df["Model"], df["Human-Rated Phonological Similarity"], color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Human-Rated Phonological Similarity")
plt.ylabel("Cosine_Similarity")
plt.ylim(0.5, 1)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
'''
plt.subplot(3, 3, 1)
bars1 = plt.bar(df_tr["Model"], df_tr["Human-Rated Phonological Similarity"], color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Human-Rated Phonological Similarity")
plt.ylabel("Cosine_Similarity")
plt.ylim(0.5, 1)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 2)
bars1 = plt.bar(df_o["Model"], df_o["Human-Rated Phonological Similarity"], color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Human-Rated Phonological Similarity")
plt.ylabel("Cosine_Similarity")
plt.ylim(0.5, 1)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 3)
bars1 = plt.bar(df_tro["Model"], df_tro["Human-Rated Phonological Similarity"], color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.title("Human-Rated Phonological Similarity")
plt.ylabel("Cosine_Similarity")
plt.ylim(0.5, 1)

for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 4)
bars2 = plt.bar(df_tr["Model"], df_tr["Orthographic Normalized Levenshtein Distance"], color="salmon")
plt.xticks(rotation=45, ha='right')
plt.title("Orthographic NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 5)
bars2 = plt.bar(df_o["Model"], df_o["Orthographic Normalized Levenshtein Distance"], color="salmon")
plt.xticks(rotation=45, ha='right')
plt.title("Orthographic NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 6)
bars2 = plt.bar(df_tro["Model"], df_tro["Orthographic Normalized Levenshtein Distance"], color="salmon")
plt.xticks(rotation=45, ha='right')
plt.title("Orthographic NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
'''
# Plot 2: Orthographic Normalized Levenshtein Distance
plt.subplot(3, 1, 2)
bars2 = plt.bar(df["Model"], df["Orthographic Normalized Levenshtein Distance"], color="salmon")
plt.xticks(rotation=45, ha='right')
plt.title("Orthographic Normalized Levenshtein Distance (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
'''
'''
# Plot 3: Phonological Normalized Levenshtein Distance
plt.subplot(3, 1, 3)
bars3 = plt.bar(df["Model"], df["Phonological Normalized Levenshtein Distance"], color="mediumseagreen")
plt.xticks(rotation=45, ha='right')
plt.title("Phonological Normalized Levenshtein Distance (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
'''
plt.subplot(3, 3, 7)
bars3 = plt.bar(df_tr["Model"], df_tr["Phonological Normalized Levenshtein Distance"], color="mediumseagreen")
plt.xticks(rotation=45, ha='right')
plt.title("Phonological NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 8)
bars3 = plt.bar(df_o["Model"], df_o["Phonological Normalized Levenshtein Distance"], color="mediumseagreen")
plt.xticks(rotation=45, ha='right')
plt.title("Phonological NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.subplot(3, 3, 9)
bars3 = plt.bar(df_tro["Model"], df_tro["Phonological Normalized Levenshtein Distance"], color="mediumseagreen")
plt.xticks(rotation=45, ha='right')
plt.title("Phonological NLD (Magnitude)")
plt.ylabel("Absolute_Value(Distance)")
plt.ylim(0.5, 1)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.tight_layout(pad=3.5)
plt.show()

# Compute the average of the three metrics
tr_indices = [0, 2, 6, 8, 12]
o_indices = [1, 3, 7, 9, 13]
tro_indices = [14, 16]
df = df.drop([4, 5, 10, 11, 15])

print(df)
new_order = [0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11]  # whatever order you want
df = df.iloc[new_order]
print(df)

df["Average_Metric"] = df[["Human-Rated Phonological Similarity", "Orthographic Normalized Levenshtein Distance", "Phonological Normalized Levenshtein Distance"]].mean(axis=1)

# Set up the grouped bar plot
labels = df["Model"]
x = np.arange(len(labels))  # label locations
width = 0.2  # width of the bars

fig, ax = plt.subplots(figsize=(14, 6))

# Plot each metric
rects1 = ax.bar(x - width, df["Human-Rated Phonological Similarity"], width, label='Human-Rated Similarity', color='skyblue')
rects2 = ax.bar(x, df["Orthographic Normalized Levenshtein Distance"], width, label='Orthographic NLD', color='salmon')
rects3 = ax.bar(x + width, df["Phonological Normalized Levenshtein Distance"], width, label='Phonological NLD', color='mediumseagreen')
rects4 = ax.bar(x + 1.5*width, df["Average_Metric"], width, label='Average Metric', color='mediumpurple')
# Labels & formatting
ax.set_ylabel("Score / Distance")
ax.set_title("Grouped Bar Plot of Model Metrics")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0.4, 1)  # Match y-axis to previous plots

fig.tight_layout()
plt.show()
