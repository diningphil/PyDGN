#!/bin/bash
#SBATCH -o slurm_cluster-%j.out
#SBATCH -e slurm_cluster-%j.err
#SBATCH -J [YOUR JOB NAME]
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node 1
#SBATCH -N 2
# with N == 2, if less than 2 nodes are available nothing starts

worker_num=1  # NOTE: Must be one less that the total number of nodes

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

export ip_head # Exporting for latter access
export redis_password

ulimit -n 65536 # increase limits to avoid to exceed redis connections

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

echo $ip_head
echo $redis_password

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

echo "<<< launcing experiment on the cluster >>>"
pydgn-train --config-file [YOUR EXP CONFIG FILE]



