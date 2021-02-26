# NTU_ADL_HW
The record of ADL course in NTU
## HW0
We use the MLP network to predict the sentence meaning.<br>
if output label is 0, it means that the sentence has negitive meaning.  
else if output label is 1, it means that the sentence has positive meaning.  
## Setup
Python 3 dependencies:
* pytorch-gpu
* pandas
* numpy
Run
```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install pandas
conda install numpy
```
## Data download
```
bash download_data.sh
```
## How to Run
open Anaconda Prompt(Anaconda3)<br>
Run the command below:
```
activate <environment_name>
cd /d<direction>
python kaggle_submission.py
```


