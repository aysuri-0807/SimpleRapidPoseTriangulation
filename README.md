# SimpleRapidPoseTriangulation
A simpler implementation of the 3D stage demonstrated in the RapidPoseTriangulation paper. 

## Process 

We follow this pipeline:

1. Score Candidates 
2. Determine Matches 
3. Reconstruct 3D objects based on pairs/matches


We start by creating tuples of detection candidates, then pass through a model to keep only the lowest error candidate pairs. 
These low error pairs are more likely to be the same object detected from different angles, thus we can use the paired data to make a 3D reconstruction. 

## Repo Setup

We define a simple `dependencies.py` script which houses the necessary functions for the pipeline. The pipeline itself is defined in `main.py`, we also include
a `test_pipeline.py` under `/tests` and store outputs in `/outputs`

## Running instructions

To run the program, instantiate a Python VENV > 3.10 and run `pip install -r requirements.txt` before running the pipeline. 