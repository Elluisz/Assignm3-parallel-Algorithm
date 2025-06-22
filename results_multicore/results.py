import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Load raw CSVs
df_seq = pd.read_csv('timing_results_sequential.csv')
df_omp = pd.read_csv('timing_results_OpenMP.csv')
df_mpi = pd.read_csv('timing_results_mpi.csv')

# Rename columns for uniformity
df_seq.columns = ['Label', 'Time (s)']
df_omp.columns = ['Label', 'Time (s)']
df_mpi.columns = ['Label', 'Time (s)']

# Add method labels
df_seq['Method'] = 'Sequential'
df_omp['Method'] = 'OpenMP'
df_mpi['Method'] = 'MPI'

# Combine into one DataFrame
df_all = pd.concat([df_seq, df_omp, df_mpi], ignore_index=True)

# Define custom colors
custom_colors = {
    'Sequential': '#1f77b4',  # blue
    'OpenMP': '#2ca02c',      # green
    'MPI': '#d62728'          # red
}

# --- Boxplot ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='Method', y='Time (s)', data=df_all, palette=custom_colors)
plt.title('Sparse Matrix Multiplication Timing Comparison')
plt.ylabel('Time (seconds)')
plt.xlabel('Method')
plt.grid(True)
plt.savefig("timing_comparison_boxplot_colored.png")
plt.close()

# --- Line Plot with Annotations ---
plt.figure(figsize=(12, 6))
for method, group in df_all.groupby("Method"):
    y = group['Time (s)'].values
    x = list(range(len(y)))
    plt.plot(x, y, label=method, color=custom_colors[method], marker='o', linewidth=1)

    min_idx = y.argmin()
    max_idx = y.argmax()
    plt.annotate(f"Min: {y[min_idx]:.4f}", (min_idx, y[min_idx]), textcoords="offset points", xytext=(0, 8), ha='center', color=custom_colors[method])
    plt.annotate(f"Max: {y[max_idx]:.4f}", (max_idx, y[max_idx]), textcoords="offset points", xytext=(0, -15), ha='center', color=custom_colors[method])

plt.title('Timing per Trial for Each Method')
plt.ylabel('Time (seconds)')
plt.xlabel('Trial')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("timing_comparison_lineplot_colored.png")
plt.close()

# --- Speedup Calculation ---
avg_seq = df_seq['Time (s)'].mean()
avg_omp = df_omp['Time (s)'].mean()
avg_mpi = df_mpi['Time (s)'].mean()

speedup_df = pd.DataFrame({
    'Method': ['OpenMP', 'MPI'],
    'Avg Time (s)': [avg_omp, avg_mpi],
    'Speedup (vs Sequential)': [avg_seq / avg_omp, avg_seq / avg_mpi]
})
speedup_df.to_csv("speedup_metrics.csv", index=False)

# --- Bar Plot of Speedups ---
plt.figure(figsize=(8, 6))
bars = plt.bar(speedup_df['Method'], speedup_df['Speedup (vs Sequential)'],
               color=[custom_colors[m] for m in speedup_df['Method']])
plt.title('Speedup Compared to Sequential')
plt.ylabel('Speedup Factor')
plt.grid(True, axis='y')

for bar, speedup in zip(bars, speedup_df['Speedup (vs Sequential)']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{speedup:.2f}", ha='center', va='bottom')

plt.savefig("speedup_barplot.png")
plt.close()

# --- Confidence Interval Plot ---
def mean_ci(series, confidence=0.95):
    mean = np.mean(series)
    sem = stats.sem(series)
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(series)-1)
    return mean, ci

ci_results = []
for df, label in [(df_seq, 'Sequential'), (df_omp, 'OpenMP'), (df_mpi, 'MPI')]:
    mean, ci = mean_ci(df['Time (s)'])
    ci_results.append({'Method': label, 'Mean': mean, 'CI': ci})

df_ci = pd.DataFrame(ci_results)

plt.figure(figsize=(8, 6))
plt.bar(df_ci['Method'], df_ci['Mean'], yerr=df_ci['CI'], capsize=5,
        color=[custom_colors[m] for m in df_ci['Method']])
plt.title('Mean Execution Time with 95% Confidence Intervals')
plt.ylabel('Time (seconds)')
plt.grid(True, axis='y')
plt.savefig("confidence_interval_barplot.png")
plt.close()
