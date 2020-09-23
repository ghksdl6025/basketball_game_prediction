# File description
## File tree 
```
📦Final project
 ┣ 📂img
 ┃ ┣ 📜crnn_loss_curve.png
 ┃ ┣ 📜gru model.png
 ┃ ┣ 📜gru_loss_curve.png
 ┃ ┣ 📜input2.png
 ┃ ┣ 📜nba_court.png
 ┃ ┣ 📜output equation.gif
 ┃ ┗ 📜timepoint equation.gif
 ┣ 📜codebook.md
 ┣ 📜CRNN_training.ipynb
 ┣ 📜Dataset check.ipynb
 ┣ 📜eventsummary.py
 ┣ 📜figuringevent.py
 ┣ 📜GRU_training.ipynb
 ┣ 📜locationtopng.py
 ┣ 📜Preprocessing.ipynb
 ┗ 📜Readme.md
```

## Data collection

[figuringevent.py](./figuringevent.py) : Downloaded game dat doens't contain detail information. With request package, call event information of the game in json format  

## Data preprocessing

[eventsummary.py](./eventsummary.py): Extractind which make the goal and miss by quarter. This is for preprocessing  

[locationtopng.py](./locationtopng.py): This file is made for two purpose.  
First from json files that contains game information, convert variables location coordinates to
graph in jpg format.  Second, transform stored image in jpg format to arrays and save in pickle format.  

[Preprocessing.ipynb](./Preprocessing.ipynb): This file is for final stage of preprocessing and creating input data
that used in deep learning model training. It shows sample input data in image format and array format. While genearing
input data, resizing step is included to save memory. After resizing, data is stored in hdf5 format.

## Model training

[GRU_training.ipynb](./GRU_training.ipynb): This file contains following contents.  
Custom dataset, dataloader, GRU model, model training, plotting loss curve,
calculate accuracy and f1-score, Size of model parameters

[CRNN_training.ipynb](./CRNN_training.ipynb): This file contains following contents.  
Custom dataset, dataloader, CRNN model, model training, plotting loss curve,
calculate accuracy and f1-score, Size of model parameters

## Apendix

[Dataset_check.ipynb](./Dataset_check.ipynb): This file is to check whether input dataset are saved in properly and
expected features.