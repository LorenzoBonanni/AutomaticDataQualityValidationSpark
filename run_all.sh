#!/bin/bash
python3 project.py --dataset="metropt3" --batch_size=100 --anomaly 0 1 2 3
python3 project.py --dataset="metropt3" --batch_size=1000 --anomaly 0 1 2 3
python3 project.py --dataset="metropt3" --batch_size=10000 --anomaly 0 1 2 3
python3 project.py --dataset="household" --batch_size=100 --anomaly 0 1 2 3
python3 project.py --dataset="household" --batch_size=1000 --anomaly 0 1 2 3
python3 project.py --dataset="household" --batch_size=10000 --anomaly 0 1 2 3
python3 project.py --dataset="onlineretail" --batch_size=100 --anomaly 0 1 2 3 4 5
python3 project.py --dataset="onlineretail" --batch_size=1000 --anomaly 0 1 2 3 4 5
python3 project.py --dataset="onlineretail" --batch_size=10000 --anomaly 0 1 2 3 4 5