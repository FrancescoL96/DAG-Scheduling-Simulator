# DAG Scheduling Simulator
This python script simulates the allocation of the nodes of the orb slam tracking on any number of CPUs and GPUs using G-FL(0), EDD(1) or HEFT(2) scheduling.

Usage: sim.py FILENAME[csv] SCHEDULER(0,1,2) FRAMES(n)

Examples:

Simulates using G-FL and 3 frames, loading from "timings.csv" (Default settings)
```
sim.py
```
Simulates using G-FL and 2 frames, loading from "timings.csv"
```
sim.py 0 2
```
Simulates using EDD and 10 frames, loading from "times.csv"
```
sim.py times.csv 1 10
```