# World-Models

# World Models Implementation In Socnavenv
This is a complete implementation, in Socnavenv , of the World Models framework described by David Ha and Jürgen Schmidhuber: https://arxiv.org/abs/1803.10122



## World Models Summary

Here is a quick summary of Ha & Schmidhuber's World Models framework. The framework aims to train an agent that can perform well in virtual gaming environments. Ha & Schmidhuber's experiments were done in the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) (from [OpenAI gym](https://gym.openai.com/)), and [ViZDoom: Take Cover](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios#take-cover) environments.In our case we are reproducing this in [socnavenv](https://github.com/robocomp/gsoc22-socnavenv) environment.


World Models consists of three main components: Vision (**V**), Model (**M**), and Controller (**C**) that interact together to form an agent:


 
**V** consists of a convolutional [Variational Autoencoder (VAE)](https://arxiv.org/abs/1606.05908), which compresses frames taken from the gameplay into a latent vector *z*. **M** consists of a [Long Short-Term Memory Recurrent Neural Networks](https://arxiv.org/pdf/1909.09586.pdf), from a [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network). This RNN takes latent vectors *z* from **V** and predicts the next frame. And finally **C** is a simple single layer linear model that maps the output from **M** to actions to perform in the environment. **C** is trained using [Evolution Strategies](https://blog.openai.com/evolution-strategies/), particularly the [CMA-ES](https://arxiv.org/abs/1604.00772) algorithm.



### Setup

* `conda install numpy chainer scipy Pillow imageio numba cupy`  *(cupy if using GPU)*  
* `pip install gym`




## Training

Training and testing our world model consists in 7 steps: 1) Generating the train and test data for VAE and RNN, 2)Generating the train and test data for RNN, 3) Training the VAE to learn a compressed representation of that data, 4) Training the RNN on sequences of the RNN train data, and 4) Testing the Trained VAE 5) Testing the Trained RNN, 6) Training the Controller, and 7) Testing the Controller.

### 1. Random Rollouts
1) Generating VAE data. Data in this case is sequences of images from the environment. To gather images from 2000 rollouts, run the command 

```sh
python3 01_generate_vae_data.py  --rollouts 2000 --testrollouts 100
```
### 2. Random Rollouts
2) Generating RNN data. Data in this case is sequences of images from the environment. To gather images from 2000 episodes, run the command 

```sh
python3 02_generate_rnn_data.py  --episodes 2000 --testepisodes 100
```
### 3. Vision (V)
3) Training the VAE. To train on our data, you can run the command

```sh
python3 03_train_vae.py --epochs 1000 --max_samples 2000
```
### 3. Model (M)
4) Training the RNN. To train on our data, you can run the command.The mode can either be window, reward, dream or normal

```sh
python3 04_train_rnn.py --epochs 1000 --mode window
```


5) Test our VAE. To test on our data, you can run the command.

```sh
python3 05_draw_vae_result.py 
```


6) Test our RNN. To test on our data, you can run the command.The mode can either be window, reward, dream or normal

```sh
python3 06_draw_rnn_result.py --mode window
```

### 7. Controller (C)

3) Training the Controller.You can run the command

```sh
python3 07_train_controller.py --epochs 100
```



## License

The original research for World Models was conducted by Ha & Schmidhuber.
