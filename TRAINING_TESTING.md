# Training and Testing Guide

This guide provides step-by-step instructions for training and testing our proposed models: the 2StepAhead Predictive World Model, Multi Action State Predictive Model (MASPM), and Action Dependent Two Step Predictive Model (ADTSPM).

## Table of Contents
- [Generating data for VAE and RNN](#generating-data-for-vae-and-rnn)
- [Training the Models](#training-the-models)
- [Testing the Models](#testing-the-models)

## Generating data for VAE and RNN
Data in this case are sequences of images from the environment. We can either use the Robot coordinate or the world coordinate.

To generate data, run:

\```sh
python3 01_generate_dataset_robot_frame.py
\```

Or:

\```sh
python3 01_generate_dataset_worldframe.py
\```

## Training the Models

### Training the VAE Model
To train the VAE on our data, run:

\```sh
python3 02_train_vae.py
\```

### Training the LSTM-RNN Model
To train the LSTM-RNN on our data, run:

\```sh
python3 05_train_rnn_main_Robotframe.py
\```

Or:

\```sh
python3 05_train_rnn_main_worldframe.py
\```

### Training the Agents
The following commands are used to train each of our proposed models:

#### MASPM
\```sh
python3 train.py -a="maspm" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/multiActionStatePredictiveModel.yaml" --kwargs run_name=maspm
\```

#### 2StepAhead MASPM 
\```sh
python3 train.py -a="2stepaheadmaspm" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/2stepaheadmultiActionStatePredictiveModel.yaml" --kwargs run_name=2stepaheadMASPM_EXP_A_SGNN
\```

#### 2StepAhead Predictive Model
\```sh
python3 train.py -a="1stepaheadpredictivemodel" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/1stepaheadpredictivemodel.yaml" --kwargs run_name=2stepaheadpredictivemodel
\```

## Testing the Models

### Testing the VAE Model
To test our VAE on our data, run:

\```sh
python3 test_vae_on_env_main.py
\```

### Testing the LSTM-RNN Model
To test our LSTM-RNN on our data, run:

\```sh
python3 test_robotframe_rnn_on_env_main.py --mode window
\```

Or:

\```sh
python3 test_worldframe_rnn_on_env_main.py --mode window
\```

### Testing the Agents
The following commands are used to test each of our proposed models:

#### MASPM
\```sh
python3 test.py -a="testmaspm" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/multiActionStatePredictiveModel.yaml" --kwargs run_name=testmaspm
\```

#### 2StepAhead MASPM Model
\```sh
python3 test.py -a="test2stepaheadmaspm" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/2stepaheadmultiActionStatePredictiveModel.yaml" --kwargs run_name=2stepaheadmaspm
\```

#### 2StepAhead Predictive Model
\```sh
python3 test.py -a="test1stepaheadpredictivemodel" -t="mlp" -e="./configs/env_timestep_1.yaml" -c="./configs/1stepaheadpredictivemodel.yaml" --kwargs run_name=test2stepaheadpredictivemodel
\```

Note: Please adjust the paths and parameters according to your exact setup and configurations.
