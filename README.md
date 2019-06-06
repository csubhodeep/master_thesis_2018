# Project Description

## Overview
This repository contains the files and scripts necessary to execute a FLISR program that utilises the tools of deep learning on an electrical distribution grid.

## Inputs
The main.py requires the scenario information and the grid topology description in csv files having a specific format.

## Algorithm
The present version of the algorithm aims to train an ANN classifier on a dataset that contains the input data of the 
power flow measurements as specified by the user and the output data as labels indicating the faulty section of the grid.

### Internal data acquisition process and data structures used
The 'data_acq.py' script is responsible to create 2D-numpy arrays with the corresponding labels

### Steps to use

1. Download the .rar from OneDrive
2. Extract the contents anywhere
3. Copy the contents of the folders 'scenario_data' and 'grid_topology'