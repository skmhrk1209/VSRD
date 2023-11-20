import argparse
import textwrap
import subprocess


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Python wrapper script over Slurm's `srun` command for multi-node single-GPU distributed data parallelism.")
    parser.add_argument("--partition", type=str, default="16gV100", help="16/32 GB Tesla V100")
    parser.add_argument("--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node you want to allocate")
    parser.add_argument("--num_gpus_per_task", type=int, default=1, help="Number of GPUs per task you want to allocate")
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes on which the processes are spawn")
    parser.add_argument("--include", type=str, default="", help="Node names on which the processes are spawn")
    parser.add_argument("--exclude", type=str, default="", help="Node names excluded from the computation resources")
    parser.add_argument("--job_name", type=str, default="", help="Job name")
    known_args, unknown_args = parser.parse_known_args()

    # NOTE: `--gpus-per-node` and `--gpus-per-task` are not supported in the current version
    num_tasks_per_node = known_args.num_gpus_per_node // known_args.num_gpus_per_task

    command = textwrap.dedent(f"""\
        srun \
            --mpi=pmi2 \
            --partition={known_args.partition} \
            --gres=gpu:{known_args.num_gpus_per_node} \
            --ntasks-per-node={num_tasks_per_node} \
            --nodes={known_args.num_nodes} \
            --nodelist={known_args.include} \
            --exclude={known_args.exclude} \
            --job-name={known_args.job_name} \
            python -u {' '.join(unknown_args)}
    """)

    subprocess.run(command, shell=True, check=False, capture_output=False)
