The directory contains the main preprocessing code for SWaT and WADI.

### SWaT
1. convert SWaT_Dataset_Normal_v0/SWaT_Dataset_Attack_v0.xlsx to csv, rename 'Normal' / 'Normal/Attack' columns to 'attack' with label 0/1
2. rename them with 'swat_train.csv' and 'swat_test.csv' as the input file in the process_swat.py
3. run the script `python process_swat.py`

### WADI
1. Based on the attack description document, add WADI_attackdata.csv with 'attack' column with 0/1 and rename file as 'WADI_attackdata_labelled.csv'
2. run the script `python process_wadi.py`

### Others
We have provided part of the processed data via [link](https://drive.google.com/drive/folders/1_4TlatKh-f7QhstaaY7YTSCs8D4ywbWc?usp=sharing).