import json
import os
import pandas as pd
import subprocess


df = pd.read_csv("wandb.csv")


print(len(df))

for row in df.itertuples():
    output_dir = row.output_dir
    t = row.Name.split("/")
    model_name, experiment = t[1], t[2]
    head = json.loads(row[7])[0]
    loss = head["loss_fns"][0]["name"]

    results_dir = f"results/{model_name}/{experiment}/{loss}"
    os.makedirs(results_dir, exist_ok=True)

    # skip if config.yaml is already in the results_dir
    if os.path.exists(os.path.join(results_dir, "config.yaml")):
        continue

    # from mpi:<outdir>/wandb/latest-run/files/config.yaml if not found continue
    s = subprocess.run(
        f"scp mpi:{output_dir}/wandb/latest-run/files/config.yaml {results_dir}/config.yaml",
        shell=True,
    )
    if s.returncode != 0:
        print(f"Failed to copy config.yaml from {output_dir}")
        continue

    # Copy files using scp from scp mpi:<outdir>/test_outputs... to results/<model_name>/<experiment>
    # Copy outputs.csv, targets.csv
    subprocess.run(
        f"scp mpi:{output_dir}/test_outputs/outputs.csv {results_dir}/outputs.csv",
        shell=True,
    )
    subprocess.run(
        f"scp mpi:{output_dir}/test_outputs/targets.csv {results_dir}/targets.csv",
        shell=True,
    )
