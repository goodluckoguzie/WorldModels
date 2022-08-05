import numpy 
import torch
import torch.nn as nn
from RNN.RNN import RNN
import time
import os, time, datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

latents = 31
actions = 2
hiddens = 256
epochs = 20

x = torch.load('./data/saved1_rollout_rnn.pt')# our training dataset got from extract_data_for_rnn.py . note that the time step here and there must tally 

class MDN_Dataset(torch.utils.data.Dataset):
    def __init__(self, MDN_data):
        self.MDN_data = MDN_data
    def __len__(self):
        return len(self.MDN_data)

    def __getitem__(self, idx):
        data = self.MDN_data[idx]
        obs = data['obs_sequence']
        action = data['action_sequence']
        #reward = data['reward_sequence']
        return (action, obs)



train_dataset = MDN_Dataset(x)




batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

l1 = nn.L1Loss()
rnn = RNN(latents, actions, hiddens).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)


start=time.time()
best_loss = float("inf")
epoch_ = []
epoch_train_loss = []
rnn.train()
for epoch in range(1, epochs + 1):
    train_loss = 0
    for batch_idx, (action, obs) in enumerate(train_dataloader):# get a batch of timesteps seperated by episodes
        # we have 200 timesteps in an episode . 
        input = obs[:, :-1, :].to("cuda:0")  # remove the last timestep . input now has 199 time steps 
        action = action[:, :-1, :]          # remove the action taken in the last timestep . input now has 199 actions time steps 
        target = obs[:, 1:, :].to("cuda:0")  #remove the first timestep .the label contains 199 timesteps because we excluded the the fist time step
        states = torch.cat([input, action], dim=-1)
        x, _, _ = rnn(states)

        loss_rnn = l1(x, target)
        optimizer.zero_grad()
        loss_rnn.backward()
        train_loss += loss_rnn.item()
        optimizer.step()
    train_loss = train_loss/ len(train_dataset)


    if train_loss <= best_loss:
        if not os.path.exists('model'):
            os.makedirs('model')
    
        torch.save(rnn.state_dict(), './MODEL/MDN_RNN_normal.pt')
        best_loss = train_loss
 
    epoch_.append(epoch)
    epoch_train_loss.append(train_loss)

    print('EPOCH : {} Average_loss : {}'.format(epoch, train_loss))   

fig = plt.figure(figsize=(8,8))
fig.set_facecolor('white')
ax = fig.add_subplot()
ax.plot(epoch_,epoch_train_loss, label='Average loss')
ax.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.show()
end_time = time.time() - start.time()
times = str(datetime.timedelta(seconds=end_time)).split(".")
print('Finished in {0}'.format(times[0]))

