# HDDse: Implementation of the HDDse Model Using Siamese LSTM

This repository implements the method described in the ATC2020 paper [HDDse: Enabling High-Dimensional Disk State Embedding for Generic Failure Detection System of Heterogeneous Disks in Large Data Centers](https://www.usenix.org/system/files/atc20-zhang-ji.pdf). The approach leverages a Siamese LSTM architecture along with a custom Euclidean distance layer to perform tasks such as anomaly detection" or "similarity measurement". In this paper, they use this technique to detect the disk failure.

## File Structure

- **parameter.py**  
  Contains hyperparameter settings for the project. Modify these parameters to tune the model’s training and inference behavior.

- **Siamese_LSTM.py**  
  Defines the network architecture based on the Siamese LSTM design as described in the paper.

- **Eludist_loss.py**  
  Implements a custom layer for computing the Euclidean distance and the associated loss function.

- **predict.py**  
  Loads the trained model and performs predictions on new input data.

## Requirements

- **Python:** Version 3.6 or higher is recommended.
- **Dependencies:**  
  - numpy  


## Installation & Usage

### 1. Clone the Repository

```bash
git clone [<repository_url>](https://github.com/Garson-hu/HDDse.git)
cd HDDse
```

## Running the model

Use the predict.py script to load the trained model and make predictions:
```
python3 predict.py --model <path_to_model> --input <path_to_input_data>
```
