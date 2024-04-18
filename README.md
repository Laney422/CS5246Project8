# CS5246Project8
This is a repo for CS5246 Project8.<br>
Google Doc's link.<br>
https://docs.google.com/document/d/1Z9y_Wu14NuTX3WYJ3hkHq9LUI-URD0W3u9F72NofxrQ/edit?usp=sharing

To run the emotion score prediction, please firstly install the requirements.txt

Before you train the model, please first unzip the data file.

To train the Bi-LSTM for emotion score prediction, use the following command:
```
python bilstm_train.py
```

To fine-tune the bert for emotion score prediction, use the following command:
```
python fine-tuned.py
```

After successfully training the model, to get the emotion score by inference, run the following command for Bi-LSTM:
```
python biLSTM_inference.py
```

For fine-tuned bert:
```
python bert_inference.py
```
