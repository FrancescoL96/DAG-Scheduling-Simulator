# DAG Scheduling Simulator
This python script simulates the allocation of the nodes of the orb slam tracking on any number of CPUs and GPUs using either EDD or G-FL scheduling.
Usage: sim.py FILENAME[csv] DEADLINE(0,1) FRAMES(n)
Example: open timings.csv, simulate three frames and use EDD
  sim.py timings.csv 1 3
