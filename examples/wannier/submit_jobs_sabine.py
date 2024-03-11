"""
Submit multiple jobs to a slurm cluster.

To run:
    $ python submit_jobs.py

    -n: create jobs but not submit
    -d: delete previous jobs

This will generate a directory named `job_dir` with the submitting script and other
files.
"""


import os
import shutil
import sys
from pathlib import Path
from typing import Any

from minilaunch import (
    LaunchDB,
    Slurm,
    ValueIter,
    iter_dict_keys,
    write_slurm_script,
    write_job_config_script,
)
from minilaunch.utils import copy_files


def submit_a_job(
    jobname: str,
    submit_dir: str | Path,
    python_script_name: str,
    default_config: str | Path,
    update_config: dict[str, Any],
    files_to_copy: list[str | Path],
    launch_db: LaunchDB,
    default_config_keys_to_pop: list[str] = None,
):
    """
    Submit a job, via the below steps:
    1. write job config script
    2. write slurm script
    3. copy other necessary files to the submitting directory
    4. submit the job

    Args:
        jobname: name of the job
        submit_dir: directory to submit the job.
        python_script_name: name of the python script
        default_config: path to the default config file
        update_config: dict to update the default config
        files_to_copy: files to copy to the submitting directory
        launch_db: db to store the job info
        default_config_keys_to_pop: pop the values identified by the keys in the default
            config. Each key is given as a dotted str. For example, assume the config
            looks like: {'a':{'b':{'c':1, 'd': 2}, 'e': 3}},
            then default_config_keys_to_pop = ['a.b.c', 'a.e'] will result in a config
            look like: {'a':{'b':{'d': 2}}}.

    Example:
        Create the jobs submitting directories but not submit them
        $ python submit_jobs.py -n
    """
    # create submit dir
    submit_dir = Path(submit_dir).expanduser().resolve()
    if submit_dir.exists():
        if "-d" in sys.argv or "--delete" in sys.argv:
            shutil.rmtree(submit_dir)
        else:
            raise RuntimeError(
                f"Submit directory already, will not override it: {submit_dir}\n"
                "Add `-d` to delete and recreate it."
            )
    os.makedirs(submit_dir)

    ## write job config script
    final_config_name = "config_final.yaml"
    write_job_config_script(
        default_config,
        update_config,
        root=submit_dir,
        final_config_name=final_config_name,
        default_config_keys_to_pop=default_config_keys_to_pop,
    )

    write_slurm_script(
        filename=submit_dir.joinpath("submit.sh"),
        job_name=jobname,
        account="wen",
        nodes=1,
        ntasks_per_node=1,
        cpus_per_task=2,
        gpus=1,
        time="2-00:00:00",
        mem="40GB",
        conda_env="/project/wen/sadhik22/envs/schnet",
        end_generic=[
            f"python {python_script_name}",
            "rm -r processed",  # remove the processed file to save disk space
        ],
    )

    # copy files
    copy_files(files_to_copy, submit_dir)

    # add info to db
    launch_db.add_job(launch_dir=submit_dir, status="created", description=f"{jobname}")

    # UNCOMMENT the below line to submit it
    if not "-n" in sys.argv:  # only generate dir, not submit
        out = Slurm().sbatch(script="submit.sh", submit_dir=submit_dir)
        print(out)


def get_update_config(
    n_atom_basis = 30,
    r_cut=5.0,
    n_interactions=3,
    n_rbf=20,
    max_epochs=1500,
    wandb_project_name="wannier_center_predictions",
):

    all_update_config = {
        #
        #datamodule
	"datamodule":{"cutoff":r_cut,"n_atom_basis":n_atom_basis,"n_interactions":n_interactions,"n_rbf":n_rbf},
        #
        # trainer
        #
        "trainer": {
            "max_epochs": max_epochs,
            "accelerator": "gpu",
            # "trainer.gradient_clip_val": [1.0, 0.0],  # 0 means no clop
            #
            # logger
            #
            "logger.init_args.project": f"{wandb_project_name}",
            #
            # callbacks
            #
            # NOTE, since callbacks is a list, but not a dict, then no `merge` will be
            # performed by omegaconf, but `replace` will be performed.
            # So, MAKE SURE to provide all callbacks here. The ones in the default
            # config file but not here will not be included.
            "callbacks": [
                {
                    "class_path": "pytorch_lightning.callbacks.ModelSummary",
                    "init_args": {
                        "max_depth": -1,
                    },
                },
                {
                    "class_path": "pytorch_lightning.callbacks.LearningRateMonitor",
                    "init_args": {
                        "logging_interval": None,
                    },
                },
                {
                    "class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "init_args": {
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 3,
                        "verbose": False,
                    },
                },
                {
                    "class_path": "pytorch_lightning.callbacks.EarlyStopping",
                    "init_args": {
                        "monitor": "val_loss",
                        "mode": "min",
                        "patience": 400,
                        "verbose": True,
                    },
                },
            ],
        },
        #
        # optimizer
        #
        "optimizer": {
            "init_args": {
                "lr": 0.01,
                "weight_decay": 0.0,
            }
        },
        #
        # lr scheduler
        #
        # "lr_scheduler": {
        #    "init_args": {
        #        "T_max": max_epochs,
        #        "eta_min": 0.0001,
        #    }
        "lr_scheduler": {
            "init_args": {
                "patience": 100,
                "factor": 0.8,
            }
        },
    }

    return iter_dict_keys(all_update_config)


if __name__ == "__main__":
    PROJECT_NAME = "wannier_center_predictions"

    BASE_CONFIG = "/home/sadhik22/Packages/schnetpack/examples/wannier/configs/config_wannier.yaml"

    JOB_DIR = "job_dir-tmp"
    if not os.path.exists(JOB_DIR):
        os.mkdir(JOB_DIR)

    launch_db = LaunchDB(db_path=f"./{JOB_DIR}/minilaunch_db.yaml", new_db=True)


    # for trainset_size in [10, 100, 1000, 2500]:
    # for trainset_size in [10]:
    for r_cut in [4.0,5.0,6.0]:
        for n_interactions in [5,6,7,8,9,10]:
            for n_atom_basis in [16,24,32,40,48]:
                for n_rbf in [16,24,32,40,48]:
                    # generate grid search of values marked by `ValueIter`
                    all_update_config = get_update_config(
                        n_atom_basis=n_atom_basis,
                        max_epochs=1500,
                        r_cut=r_cut,
                        n_interactions=n_interactions,
                        n_rbf=n_rbf,
                        wandb_project_name=PROJECT_NAME + f"_rcut_{r_cut}_nlay_{n_interactions}_atm_bas_{n_atom_basis}_n_gaus_{n_rbf}",)
                    config=all_update_config[-1]
                    submit_a_job(jobname=f"rcut_{r_cut}_nlay_{n_interactions}_atm_bas_{n_atom_basis}_n_gaus_{n_rbf}",
                        submit_dir=f"{JOB_DIR}/rcut_{r_cut}_nlay_{n_interactions}_atm_bas_{n_atom_basis}_n_gaus_{n_rbf}",
                        python_script_name="train_wannier.py",
                        default_config=BASE_CONFIG,
                        update_config=config,
                        files_to_copy=["train_wannier.py"],
                        launch_db=launch_db,
                        default_config_keys_to_pop=None,)
