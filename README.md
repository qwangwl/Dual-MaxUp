# Dual-MaxUp (Draft Implementation)

This repository contains a **Pytorch implementation** of "Dual-Maxup: A Dual-Maximization Data Augmentation Strategy for Cross-Subject EEG Emotion Recognition in Meta-Transfer Learning". 

## Code Structure

- **`run_pre_train`**: One-click script to run pretrain
- **`run_meta_adapt`**: One-click script to run meta_adapt
- **`run_meta_test`**: One-click script to run meta_test
- **Other files**: Include model definitions, training utilities, and data processing scripts.

## Description

This codebase contains all the experiments from the paper. If you only want to apply the algorithm described in this paper, you can run `run_pre_train`, `run_meta_adapt`, and `run_meta_test` respectively to obtain the final results.

Please note that you need to change the data directory and the directory where you save the results.