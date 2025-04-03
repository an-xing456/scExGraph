import os
import pandas as pd
import sys

if len(sys.argv) != 2:
    print("Usage: data_summary.py <base_path>")
    sys.exit(1)

base_path = sys.argv[1]

print(base_path)

process_path = f"{base_path}processed/"

if not os.path.exists(process_path):
    os.makedirs(process_path)
    
output_file = os.path.join(process_path, "combined.csv")

summary_files = sorted([f for f in os.listdir(process_path) if f.endswith('_summary.csv')])

with open(output_file, 'w') as f_out:
    for i, file in enumerate(summary_files):
        file_path = os.path.join(process_path, file)
        df = pd.read_csv(file_path, index_col=0)
        
        df.reset_index(inplace=True)
        
        if i > 0:
            f_out.write('\n' * 2)

        df.to_csv(f_out, header=True, index=False)

print(f"All files have been merged and saved to {output_file}")
