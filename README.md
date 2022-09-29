# World-Models
# World Models Implementation In Socnavenv
This is a complete implementation, in Socnavenv , of the World Models framework described by David Ha and Jürgen Schmidhuber: https://arxiv.org/abs/1803.10122


## World Models Summary

Here is a quick summary of Ha & Schmidhuber's World Models framework. The framework aims to train an agent that can perform well in virtual gaming environments. Ha & Schmidhuber's experiments were done in the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) (from [OpenAI gym](https://gym.openai.com/)), and [ViZDoom: Take Cover](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios#take-cover) environments.In our case we are reproducing this in [socnavenv](https://github.com/robocomp/gsoc22-socnavenv) environment.


World Models consists of three main components: Vision (**V**), Model (**M**), and Controller (**C**) that interact together to form an agent:
 
**V** consists of a convolutional [Variational Autoencoder (VAE)](https://arxiv.org/abs/1606.05908), which compresses frames taken from the gameplay into a latent vector *z*. **M** consists of a [Long Short-Term Memory Recurrent Neural Networks](https://arxiv.org/pdf/1909.09586.pdf), from a [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network). This RNN takes latent vectors *z* from **V** and predicts the next frame. And finally **C** is a simple single layer linear model that maps the output from **M** to actions to perform in the environment. **C** is trained using [Evolution Strategies](https://blog.openai.com/evolution-strategies/), particularly the [CMA-ES](https://arxiv.org/abs/1604.00772) algorithm.


### To Train our Baseline duelingDQN 
We trained our agent using duelling DQN in order to set a benchmark for on-going research. To train our benchmark model, run the code below.

```sh
python3 T.py -a="duelingdqn" -t="mlp" -e="./configs/env.yaml" -c="./configs/duelingDQN.yaml" --kwargs run_name=dueling_mlp
```

### To Generate our Dataset 
1) To generate our dataset. The dataset are flatten episodes,generating sequence of rollouts. To generate images from 2000 rollouts, run the command below. This command generates test,train and validation dataset.

```sh
python3 01_generate_vae_dataset.py  ----rollouts 2000 
```
### Train Our VAE Model
2) To train on our model for 1000 episode run the command below
```sh
python3 02_train_vae.py --epochs 1000 
```

### Train Our RNN
5) To train on our model for 1000 episode run the command below 

```sh
python3 05_train_rnn.py --epochs 1000
```

### Train Our Controller 
7) To train on our controller model, run the command below 

```sh
python3 07_train_contoller.py --episodes 1000
```

### Test Our Controller 
8) To test on our controller model, run the command below 

```sh
python3 08_test_controller.py --episodes 1000
```


### Train Our DuelingDQN with our trained RNN and VAE model
 To train our model, run the code below.

```sh
python3 T.py -a="duelingdqn" -t="mlp" -e="./configs/env.yaml" -c="./configs/DQN_RNN_VAE.yaml" --kwargs run_name=dueling_mlp
```


## License

The original research for World Models was conducted by Ha & Schmidhuber.
