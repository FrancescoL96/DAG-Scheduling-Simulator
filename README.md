# DAG Scheduling Simulator
This python script simulates the allocation of the nodes of an OpenVX DAG on any number of CPUs and GPUs using G-FL(0), EDD(1) or HEFT(2)/XEFT(4) scheduling. A custom version of G-FL is also available(3), which uses G-FL normally for scheduling, but uses the HEFT mapping instead of the default approach (VisionWork style).
## Requirements
In order to run these scripts the following libraries are required:
```
matplotlib
scipy
```
## Usage
sim.py FILENAME[csv] SCHEDULER(0,1,2,3,4) FRAMES(n) PIPELINE(0,1)

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
### Auto simulation with generated graphs
Simulates a number of generated graphs (using generator.py and sim.py), output in "output.csv"
```
auto_sim.py
```

### Re-simulate with past simulations
Exctracts additional data (now configured for memory footprint), has to be run in the same folder as the *.pkl files generated by sim.py, outputs to "result.csv", use "merge.py" to merge with "output.csv"
```
re_sim.py
```
