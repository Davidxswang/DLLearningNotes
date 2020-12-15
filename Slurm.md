# Slurm

## Credits

Credits are given to [CECI-HPC Docs](https://support.ceci-hpc.be/doc/_contents/QuickStart/SubmittingJobs/SlurmTutorial.html) and [CECI-HPC FAQ](https://support.ceci-hpc.be/doc/_contents/SubmittingJobs/SlurmFAQ.html).

## show information

### sinfo

Example: `sinfo` or `sinfo -l -N -p partitionA` or `sinfo -o "%15N %10c %10m %25f %10G"`

This will:

- show the cluster information
- `-l` show detailed output
- `-N` to switch to node centric mode
- `-p partitionA` to show which partition
In the output, partitions with `*` is the default partition.
- `-o` will specify the output format. N, c, m, f and G, for example, stand for NodeList, CPUs, Memory, Features, Generic Resources, respectively.

### squeue

Example: `squeue -u userabc -j 123 -o %Q`

This will:

- show the jobs of the `userabc` in the queue, which are either in running or pending state.
- `-j` show the job info with jobid it specifies
- `-o %Q` show the priority of the job

### sstat

Example: `sstat -j 123`

This will:

- show the statistics of the job 123 if the job is running.


### sacct

Example: `sacct --format JobID,jobname,NTasks,nodelist,MaxRSS,MaxVMSize,AveRSS,AveVMSize`

This will:

- show the accounting information bond with the logged-in user, provided the jobs are finished.

### sprio

Example: `sprio`

This will:

- show the priorities of the pending jobs 

### sshare

Example: `sshare`

This will:

- show the user's fairshare

### scontrol

Example: `scontrol show job 123` or `scontrol show nodes`

This will:

- show the details of job 123
- show all the info of all the nodes

## create a job

### general steps

To create a job, two parts are needed:

1. resources request, using `#SBATCH` in the top of the script
2. job steps, using `srun in the bottom of the script`

### detailed steps to submit a job using a script

1. write a job script
Example: `submit.sh`

```python
#!/bin/bash
#
#SBATCH --job-name=gputest
#SBATCH --output=gpures.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --partition=gvohp

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun nvidia-smi
srun env
echo "finished"
srun sleep 30
```

In the example above,

- `--ntasks` specifies the process, not thread. A multi-process program is made of several tasks, while a multi-thread program is made of several threads within one process, which will run on several CPU cores (or CPUs in the Slurm context). One single task cannot be split across several compute nodes, so using `--ntasks=2 --cpus-per-task=4` will make sure that each one of these two tasks will get 4 CPUs, and these 4 CPUs (belonging to the same task) will be on the same node. If using `--ntasks=8` will get 8 CPUs as well, but we cannot know how many nodes these 8 CPUs will span.
- `--time` specifies the time limit we set for this job
- `--mem-per-cpu` specifies how many memory we want to allocate calculated by cpu
- `--gres` specifies what generic resource scheduling is needed, like `gpu:volta:1` or `gpu:1`
- `--qos` specifies the QoS, like gpu, primary, etc
- `--partition` specifies the partition, like gvohp, gsmtp, etc
- `OMP_NUM_THREADS` specifies the number of threads per process, so it should be the same with the argument `--cpus-per-task`
- `srun` will run the actual job steps just like what we do in the terminal

2. submit the script

Example: in the terminal run `sbatch submit.sh`

### run an interactive job

Example: `srun -t 1:00:00 -n 1 --mem=2G --pty bash`

This will:

- request the resources of 1 CPU and 2G memory
- open a bash terminal in the allocated node

### the priority of a job

1. find out the scheduler type
Example: `scontrol show config | grep Scheduler`

If SchedulerType is `sched/backfill`, the scheduler is slurm itself. Otherwise, the priority calculation is done externally by a third-party app.

2. find out the priority type if using Slurm scheduler
Example: `scontrol show config | grep Priority`

If PriorityType is `priority/multifactor`, the priority depends on 5 elements:

- Job age
- User fairshare
- Job size
- Partition
- QoS

Using the above command, we can see all the priority weight like `PriorityWeight****`.

## cancal a job

### scancel

Example: `scancel 123`

This will:

- cancel the job with id 123
