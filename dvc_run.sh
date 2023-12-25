#!/bin/bash
dvc init 
dvc pull
dvc stage add --name train --outs dvclive python3 train.py
dvc exp run