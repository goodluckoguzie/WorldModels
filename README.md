# World-Models
# World Models Implementation In Socnavenv
This is a complete implementation, in Socnavenv , of the World Models framework described by David Ha and Jürgen Schmidhuber: https://arxiv.org/abs/1803.10122


## World Models Summary

Here is a quick summary of Ha & Schmidhuber's World Models framework. The framework aims to train an agent that can perform well in virtual gaming environments. Ha & Schmidhuber's experiments were done in the [CarRacing-v0](https://gym.openai.com/envs/CarRacing-v0/) (from [OpenAI gym](https://gym.openai.com/)), and [ViZDoom: Take Cover](https://github.com/mwydmuch/ViZDoom/tree/master/scenarios#take-cover) environments.In our case we are reproducing this in [socnavenv](https://github.com/robocomp/gsoc22-socnavenv) environment.


World Models consists of three main components: Vision (**V**), Model (**M**), and Controller (**C**) that interact together to form an agent:
 
**V** consists of a convolutional [Variational Autoencoder (VAE)](https://arxiv.org/abs/1606.05908), which compresses frames taken from the gameplay into a latent vector *z*. **M** consists of a [Long Short-Term Memory Recurrent Neural Networks](https://arxiv.org/pdf/1909.09586.pdf), from a [Recurrent Neural Network (RNN)](https://en.wikipedia.org/wiki/Recurrent_neural_network). This RNN takes latent vectors *z* from **V** and predicts the next frame. And finally **C** is a simple single layer linear model that maps the output from **M** to actions to perform in the environment. **C** is trained using [Evolution Strategies](https://blog.openai.com/evolution-strategies/), particularly the [CMA-ES](https://arxiv.org/abs/1604.00772) algorithm.



### VAE Dataset 
1) To generate our dataset for VAE. The dataset are flatten episodes,generating sequence of rollouts. To generate images from 2000 rollouts, run the command below. This command generates test,train and validation dataset.

```sh
python3 01_generate_vae_dataset.py  ----rollouts 2000 
```
### Train Our VAE Model
2) Training the VAE. To train on our data for 1000 episode run the command below
```sh
python3 02_train_vae.py --epochs 1000 
```



### Test Our VAE
3) After training our VAE model. To test on our model, run the command below

```sh
python3 03_test_vae.py 
```
### RNN Dataset
4) To generate a dataset of 1000 episodes, run the command below. 

```sh
python3 04_generate_rnn_dataset.py --epochs 1000 -
```





## License

The original research for World Models was conducted by Ha & Schmidhuber.
