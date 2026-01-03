Radar UAV Target Classification

This repository contains the code and final report for a radar-based multi-class target classification project.
The objective is to classify radar targets into Cars, Drones, and People using deep learning models trained on the Real Doppler RAD-DAR dataset.

The work focuses on comparing recurrent and attention-based neural architectures under a unified preprocessing, training, and evaluation pipeline.

Project Overview

Radar time–frequency representations (micro-Doppler maps) are used as inputs to the models.
Each radar sample is represented as a 128x128 two-dimensional map and assigned to one of three target classes.

The following model architectures are implemented and evaluated:

- CNN + BiLSTM, used as the recurrent baseline
- CNN + xLSTM, implemented in two stabilized variants
- CNN + TransformerEncoder, used for global attention-based modeling

Model performance is evaluated using accuracy, macro-F1 score, per-class precision, recall, F1-score, and confusion matrices.

Repository Structure

- raddar_part1_bilstm.py  
  CNN and BiLSTM based radar classification model

- raddar_part2_xlstmcnn.py  
  CNN and xLSTM model (variant 1)

- raddar_part3_xlstmcnnv2.py  
  CNN and xLSTM model (variant 2)

- raddarpart4_transformercnn.py  
  CNN and Transformer-based radar classification model

- saadet-busra-cam-21-final.pdf  
  Final project report including methodology, experiments, and analysis

Dataset

The project uses the Real Doppler RAD-DAR database.
Radar measurements are stored as CSV files representing time–frequency intensity maps.

Classes:
- Cars
- Drones
- People

The dataset is split into training, validation, and test sets using a stratified 70/15/15 ratio.
Balanced subsets are used to ensure fair evaluation across classes.

Dataset source:
https://www.kaggle.com/datasets/iroldan/real-doppler-raddar-database


Requirements

The code is written in Python and relies on the following main libraries:

- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Exact versions can be inferred from the source files if needed.


Running the Code

Each model can be trained and evaluated independently by running the corresponding script:

python raddar_part1_bilstm.py  
python raddar_part2_xlstmcnn.py  
python raddar_part3_xlstmcnnv2.py  
python raddarpart4_transformercnn.py  

Training logs, loss and accuracy curves, and confusion matrices are produced during execution.


Results Summary

All evaluated models achieve strong performance on the balanced RAD-DAR dataset.
The People class is consistently classified with high accuracy.
The primary source of confusion occurs between the Cars and Drones classes due to similarities in their radar motion signatures.

The xLSTM variant 1 achieves the highest overall accuracy and macro-F1 score.
The Transformer-based model provides stable training dynamics and strong global context modeling but does not surpass the best-performing recurrent model in this experimental setting.

Detailed quantitative results and discussions are provided in the final report.

Report

The file contains the complete project report, including:
- theoretical background
- model architecture descriptions
- training setup
- experimental results
- discussion and conclusions


Authors

Saadet Büşra ÇAM & Bilge Şentürk
EE443 Neural Networks  
Bilkent University
