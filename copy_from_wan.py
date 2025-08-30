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

    # from mpi:<outdir>/wandb/latest-run/files/config.yaml
    subprocess.run(
        f"scp mpi:{output_dir}/wandb/latest-run/files/config.yaml {results_dir}/config.yaml",
        shell=True,
    )
