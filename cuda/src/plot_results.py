import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
CSV_FILE = 'data/runtime_data.csv'
RESULTS_DIR = 'results'

if not os.path.exists(CSV_FILE):
    print(f"Error: CSV file not found at {CSV_FILE}.")
    print("Please ensure you have run the compiled C++ executable first to generate the data.")
    sys.exit(1)

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

df['Time_s'] = df['Time_ms'] / 1000.0
df_grouped = df.groupby(['MatrixSize', 'Version'])['Time_s'].mean().reset_index()
pivot_df = df_grouped.pivot(index='MatrixSize', columns='Version', values='Time_s')

# Get the CPU Multi-Threaded time series for the speedup baseline
if 'CPU_V1_Multi' not in pivot_df.columns:
    print("Error: CPU_V1_Multi data not found in CSV. Cannot calculate speedup.")
    sys.exit(1)
cpu_multi_time = pivot_df['CPU_V1_Multi']


# --- Calculations: Speedup Relative to CPU Multi-Threaded (V1) ---
pivot_df['Speedup_V2_Naive'] = cpu_multi_time / pivot_df['CUDA_V2_Naive']
pivot_df['Speedup_V3_Coalesced'] = cpu_multi_time / pivot_df['CUDA_V3_Coalesced']
pivot_df['Speedup_V4_SharedTiled'] = cpu_multi_time / pivot_df['CUDA_V4_SharedTiled']
pivot_df['Speedup_V5_AsyncStreams'] = cpu_multi_time / pivot_df['CUDA_V5_AsyncStreams']
pivot_df['Speedup_V6_PinnedAsync'] = cpu_multi_time / pivot_df['CUDA_V6_PinnedAsync']


# --- 2. Plotting Execution Time (Log Scale) ---
plt.figure(figsize=(12, 7))
plot_versions = ['CPU_V0_Single', 'CPU_V1_Multi', 'CUDA_V2_Naive', 'CUDA_V3_Coalesced', 'CUDA_V4_SharedTiled', 'CUDA_V5_AsyncStreams', 'CUDA_V6_PinnedAsync']
labels = ['CPU V0: Single', 'CPU V1: Multi (OpenMP)', 'CUDA V2: Naive', 'CUDA V3: Coalesced', 'CUDA V4: Shared Tiled', 'CUDA V5: Async Streams', 'CUDA V6: Pinned + Streams']

for version, label in zip(plot_versions, labels):
    if version in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[version], marker='o', linestyle='-', label=label)

plt.title('Execution Time Comparison (Log Scale)', fontsize=16)
plt.xlabel('Matrix Size (N x N)', fontsize=14)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(title="Implementation Version", loc='upper left', fontsize=10)
plt.xticks(pivot_df.index, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '01_Execution_Time_7_Versions.png'))
# plt.show()


# --- 3. Plotting Speedup Relative to CPU Multi-Threaded ---
plt.figure(figsize=(12, 7))
speedup_cols = ['Speedup_V2_Naive', 'Speedup_V3_Coalesced', 'Speedup_V4_SharedTiled', 'Speedup_V5_AsyncStreams', 'Speedup_V6_PinnedAsync']
speedup_labels = ['V2: Naive', 'V3: Coalesced', 'V4: Shared Tiled', 'V5: Async Streams', 'V6: Pinned + Streams']

for col, label in zip(speedup_cols, speedup_labels):
    if col in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[col], marker='o', linestyle='-', label=label)

plt.title('Speedup Factor Relative to CPU Multi-Threaded (OpenMP)', fontsize=16)
plt.xlabel('Matrix Size (N x N)', fontsize=14)
plt.ylabel('Speedup Factor', fontsize=14)
plt.axhline(1.0, color='r', linestyle='--', linewidth=1, label='Break-Even (Speedup=1)')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(title="GPU Optimization Level", loc='upper left', fontsize=10)
plt.xticks(pivot_df.index, rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, '02_Speedup_vs_CPU_Multi.png'))
# plt.show()


print(f"\n--- Analysis Complete ---")
print(f"Raw data is in: {CSV_FILE}")
print(f"Graphs are saved in the '{RESULTS_DIR}/' directory.")
print("\nMax Speedup of each GPU version relative to CPU Multi-threaded:")
print(pivot_df[['Speedup_V2_Naive', 'Speedup_V3_Coalesced', 'Speedup_V4_SharedTiled', 'Speedup_V5_AsyncStreams', 'Speedup_V6_PinnedAsync']].max())