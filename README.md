# DAG Scheduling Simulator
This python script simulates the allocation of the nodes of the orb slam tracking on any number of CPUs and GPUs using G-FL(0), EDD(1) or HEFT(2) scheduling.

Usage: sim.py FILENAME[csv] SCHEDULER(0,1,2) FRAMES(n) PIPELINE(0,1)

Examples:

Simulates using G-FL , 3 frames and no pipeline, loading from "graph_file.csv" (Default settings)
```
sim.py
```
Simulates using EDD and 2 frames, loading from "graph_file.csv", using the pipeline
```
sim.py 1 2 1
```
Simulates using HEFT, 10 frames and pipeline, loading from "graph.csv"
```
sim.py graph.csv 2 10 1
```