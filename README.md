# Learning MIPT with QuAN
This is the code repository for "Learning measurement-induced phase transition using attention".

Due to the large volume, `Data_src` folder is not present in this repository. It is provided upon reasonable request.
- `1-StateDistinguishing`: Training and testing results for state distinguishing task (learnability transition.)
- `2-PhaseRecognition`: Training and testing results for phase recognition task. Folder includes inter-trajectory attention score data extraction.
- `Data_out` folder contains training outcomes. `Fig_*.ipynb` files analyze results and generate figure for the main and supplementary text.

## 1-StateDistinguishing
### Training
`model_name` ranges from `QuAN4_Tbt2` (full model), `SMLP_Tbt2` (temporal attention only), `SMLP_Tbm2` (no attention). System size are even number between 8 and 20. Sample size should be smaller or equal to 6,000,000. Monitoring strength is elaborated in the main text of the paper.
```
bash train_statedistinguishing.sh "model_name" "system size L" "set size N" "sample size M" "monitoring strength gamma"
```

### Testing
```
bash test_statedistinguishing.sh "model_name" "system size L" "set size N" "sample size M" "monitoring strength gamma"
```

## 2-PhaseRecognition
### Training
`model_name` ranges from `QuAN4_Tbt2` (full model), `SMLP_Tbt2` (temporal attention only), `SMLP_Tbm2` (no attention). System size are even number between 8 and 20. Sample size should be smaller or equal to 40,000. Monitoring strength is elaborated in the main text of the paper.
```
bash train_phaserecognition.sh "model_name" "system size L" "set size N" "sample size M"
```

### Testing
```
bash test_phaserecognition.sh "model_name" "system size L" "set size N" "sample size M"
bash test_attn.sh "model_name" "system size L" "set size N" "sample size M"
```