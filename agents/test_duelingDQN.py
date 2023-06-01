import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import socnavenv
import gym
import numpy as np
import copy
import random
import torch.optim as optim
import os
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from agents.models import MLP, ExperienceReplay


class DuelingDQN(nn.Module):
    def __init__(self, input_size, hidden_layers: list, v_net_layers: list, a_net_layers: list) -> None:
        super().__init__()
        # sizes of the first layer in the value and advantage networks should be same as the output of the hidden layer network
        assert(v_net_layers[0] == hidden_layers[-1]
               and a_net_layers[0] == hidden_layers[-1])
        self.hidden_mlp = MLP(input_size, hidden_layers)
        self.value_network = MLP(v_net_layers[0], v_net_layers[1:])
        self.advantage_network = MLP(a_net_layers[0], a_net_layers[1:])

    def forward(self, x):
        x = self.hidden_mlp.forward(x)
        v = self.value_network.forward(x)
        a = self.advantage_network.forward(x)
        q = v + a - torch.mean(a, dim=1, keepdim=True)
        return q


class DuelingDQNAgent:
    def __init__(self, env: gym.Env, config: str, **kwargs) -> None:
        assert(env is not None and config is not None)
        # initializing the env
        self.env = env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # agent variables
        self.input_layer_size = None
        self.hidden_layers = None
        self.v_net_layers = None
        self.a_net_layers = None
        self.buffer_size = None
        self.num_episodes = None
        self.epsilon = None
        self.epsilon_decay_rate = None
        self.batch_size = None
        self.gamma = None
        self.lr = None
        self.polyak_const = None
        self.render = None
        self.min_epsilon = None
        self.save_path = None
        self.render_freq = None
        self.save_freq = None
        self.run_name = None

        # if variables are set using **kwargs, it would be considered and not the config entry
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise NameError(f"Variable named {k} not defined")

        # setting values from config file
        self.configure(self.config)

        # declaring the network
        self.duelingDQN = DuelingDQN(
            self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)

        # initializing the fixed targets
        self.fixed_targets = DuelingDQN(
            self.input_layer_size, self.hidden_layers, self.v_net_layers, self.a_net_layers).to(self.device)
        self.fixed_targets.load_state_dict(self.duelingDQN.state_dict())

        # initalizing the replay buffer
        self.experience_replay = ExperienceReplay(self.buffer_size)

        # variable to keep count of the number of steps that has occured
        self.steps = 0

        if self.run_name is not None:
            self.writer = SummaryWriter('runs/'+self.run_name)
        else:
            self.writer = SummaryWriter()

    def configure(self, config: str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.input_layer_size is None:
            self.input_layer_size = config["input_layer_size"]
            assert(
                self.input_layer_size is not None), f"Argument input_layer_size cannot be None"

        if self.hidden_layers is None:
            self.hidden_layers = config["hidden_layers"]
            assert(
                self.hidden_layers is not None), f"Argument hidden_layers cannot be None"

        if self.v_net_layers is None:
            self.v_net_layers = config["v_net_layers"]
            assert(
                self.v_net_layers is not None), f"Argument v_net_layers cannot be None"

        if self.a_net_layers is None:
            self.a_net_layers = config["a_net_layers"]
            assert(
                self.a_net_layers is not None), f"Argument a_net_layers cannot be None"

        if self.buffer_size is None:
            self.buffer_size = config["buffer_size"]
            assert(self.buffer_size is not None), f"Argument buffer_size cannot be None"

        if self.num_episodes is None:
            self.num_episodes = config["num_episodes"]
            assert(
                self.num_episodes is not None), f"Argument num_episodes cannot be None"

        if self.epsilon is None:
            self.epsilon = config["epsilon"]
            assert(self.epsilon is not None), f"Argument epsilon cannot be None"

        if self.epsilon_decay_rate is None:
            self.epsilon_decay_rate = config["epsilon_decay_rate"]
            assert(
                self.epsilon_decay_rate is not None), f"Argument epsilon_decay_rate cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.gamma is None:
            self.gamma = config["gamma"]
            assert(self.gamma is not None), f"Argument gamma cannot be None"

        if self.lr is None:
            self.lr = config["lr"]
            assert(self.lr is not None), f"Argument lr cannot be None"

        if self.polyak_const is None:
            self.polyak_const = config["polyak_const"]
            assert(
                self.polyak_const is not None), f"Argument polyak_const cannot be None"

        if self.render is None:
            self.render = config["render"]
            assert(self.render is not None), f"Argument render cannot be None"

        if self.min_epsilon is None:
            self.min_epsilon = config["min_epsilon"]
            assert(self.min_epsilon is not None), f"Argument min_epsilon cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.render_freq is None:
            self.render_freq = config["render_freq"]
            assert(self.render_freq is not None), f"Argument render_freq cannot be None"

        if self.save_freq is None:
            self.save_freq = config["save_freq"]
            assert(self.save_freq is not None), f"Argument save_freq cannot be None"

    def xavier_init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    # def preprocess_observation(self, obs):
    #     """
    #     To convert dict observation to numpy observation
    #     """
    #     assert(type(obs) == dict)
    #     observation = np.array([], dtype=np.float32)
    #     observation = np.concatenate((observation, obs["goal"].flatten()) )
    #     observation = np.concatenate((observation, obs["humans"].flatten()) )
    #     observation = np.concatenate((observation, obs["laptops"].flatten()) )
    #     observation = np.concatenate((observation, obs["tables"].flatten()) )
    #     observation = np.concatenate((observation, obs["plants"].flatten()) )
    #     return observation
    def preprocess_observation(self, obs):
        """
        To convert dict observation to numpy observation
        """
        assert(type(obs) == dict)
        obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
        humans = obs["humans"].flatten()
        for i in range(int(round(humans.shape[0]/(6+7)))):
            index = i*(6+7)
            obs2 = np.concatenate((obs2, humans[index+6:index+6+7]))
        # return torch.from_numpy(obs2)
        return obs2

    def discrete_to_continuous_action(self, action: int):
        """
        Function to return a continuous space action for a given discrete action
        """
        if action == 0:
            return np.array([0, 1], dtype=np.float32)
        # Turning clockwise
        elif action == 1:
            return np.array([0, -1], dtype=np.float32)
        # Turning anti-clockwise and moving forward
        # elif action == 3:
        #     return np.array([1, 0.5], dtype=np.float32)
        # # Turning clockwise and moving forward
        # elif action == 4:
        #     return np.array([1, -0.5], dtype=np.float32)
        # # Move forward
        elif action == 2:
            return np.array([1, 0], dtype=np.float32)
        # stop the robot
        elif action == 3:
            return np.array([0, 0], dtype=np.float32)
            # Turning clockwise with a reduced speed and rotation
        # elif action == 7:
        #     return np.array([0.5, 1], dtype=np.float32)
        #     # Turning anti-clockwise with a reduced speed and rotation
        # elif action == 8:
        #     return np.array([0.5, -1], dtype=np.float32)

        else:
            raise NotImplementedError

    def get_action(self, current_state, epsilon):

        # exploit
        with torch.no_grad():
            q = self.duelingDQN(torch.from_numpy(
                current_state).reshape(1, -1).float().to(self.device))
            action_discrete = torch.argmax(q).item()
            action_continuous = self.discrete_to_continuous_action(
                action_discrete)
            return action_continuous, action_discrete

    def calculate_grad_norm(self):
        total_norm = 0
        parameters = [p for p in self.duelingDQN.parameters(
        ) if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def save_model(self, path):
        torch.save(self.duelingDQN.state_dict(), path)

    def update(self):
        curr_state, rew, act, next_state, d = self.experience_replay.sample_batch(
            self.batch_size)

        # a_max represents the best action on the next state according to the original network (the network other than the target network)
        a_max = torch.argmax(self.duelingDQN(torch.from_numpy(
            next_state).float().to(self.device)), keepdim=True, dim=1)

        # calculating target value given by r + (gamma * Q(s', a_max, theta')) where theta' is the target network parameters
        # if the transition has done=True, then the target is just r

        # the following calculates Q(s', a) for all a
        q_from_target_net = self.fixed_targets(
            torch.from_numpy(next_state).float().to(self.device))

        # calculating Q(s', a_max) where a_max was the best action calculated by the original network
        q_s_prime_a_max = torch.gather(
            input=q_from_target_net, dim=1, index=a_max)

        # calculating the target. The above quantity is being multiplied element-wise with ~d, so that only the episodes that do not terminate contribute to the second quantity in the additon
        target = torch.from_numpy(rew).float().to(self.device) + self.gamma * (
            q_s_prime_a_max * (~torch.from_numpy(d).bool().to(self.device)))

        # the prediction is given by Q(s, a). calculting Q(s,a) for all a
        q_from_net = self.duelingDQN(torch.from_numpy(
            curr_state).float().to(self.device))

        # converting the action array to a torch tensor
        act_tensor = torch.from_numpy(act).long().to(self.device)

        # calculating the prediction as Q(s, a) using the Q from q_from_net and the action from act_tensor
        prediction = torch.gather(input=q_from_net, dim=1, index=act_tensor)

        # loss using MSE
        loss = self.loss_fn(prediction, target)
        self.episode_loss += loss.item()

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        self.total_grad_norm += torch.nn.utils.clip_grad_norm_(
            self.duelingDQN.parameters(), max_norm=0.5).cpu()
        self.optimizer.step()

    def plot(self, episode):
        self.rewards.append(self.episode_reward)
        self.losses.append(self.episode_loss)
        self.exploration_rates.append(self.epsilon)
        self.grad_norms.append(self.total_grad_norm/self.batch_size)
        self.successes.append(self.has_reached_goal)
        self.collisions.append(self.has_collided)
        self.steps_to_reach.append(self.steps)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))

        np.save(os.path.join(self.save_path, "plots", "rewards"),
                np.array(self.rewards), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "losses"), np.array(
            self.episode_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "exploration_rates"),
                np.array(self.epsilon), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(
            self.total_grad_norm/self.batch_size), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "successes"), np.array(
            self.has_reached_goal), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "collisions"), np.array(
            self.has_collided), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "steps_to_reach"),
                np.array(self.steps), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar(
            "reward / epsiode", self.episode_reward, episode)
        self.writer.add_scalar("loss / episode", self.episode_loss, episode)
        self.writer.add_scalar(
            "exploration rate / episode", self.epsilon, episode)
        self.writer.add_scalar("Average total grad norm / episode",
                               (self.total_grad_norm/self.batch_size), episode)
        self.writer.add_scalar("ending in sucess? / episode",
                               self.has_reached_goal, episode)
        self.writer.add_scalar("has collided? / episode",
                               self.has_collided, episode)
        self.writer.add_scalar(
            "Steps to reach goal / episode", self.steps, episode)
        self.writer.flush()

    def eval(self, num_episodes=500, path=None):
        if path is None:
            path = os.getcwd()

            self.duelingDQN.load_state_dict(torch.load(
                './models/duelingDQN_input_23_512_128_Exp_1/episode00198250.pth'))
            # self.duelingDQN.load_state_dict(torch.load('./models/duelingDQN_input_23_512_128_Exp_2/episode00078850.pth'))
            # self.duelingDQN.load_state_dict(torch.load('./models1/duelingDQN_input_23_512_128_Exp_3/episode00058650.pth'))

        self.duelingDQN.eval()

        # total_reward = 0
        reward_per_episode = 0
        successive_runs = 0

        total_jerk_count = 0
        total_velocity = np.array([0.0, 0.0])
        total_path_length = 0
        total_time = 0
        total_idle_time = 0

        # Add counters for the conditions
        out_of_map_count = 0
        human_collision_count = 0
        reached_goal_count = 0
        max_steps_count = 0
        discomfort_count = 0
        personal_space_radius = 2.02  # Define your personal space radius

        # Initialize lists to store values for each episode
        discomfort_counts = []
        jerk_counts = []
        velocities = []
        path_lengths = []
        times = []
        out_of_maps = []
        human_collisions = []
        reached_goals = []
        max_steps = []
        episode_run = []
        successive_run = []
        episode_reward = []
        idle_times = []
        personal_space_compliances = []

        def transform_processed_observation_into_raw(sample):

            ind_pos = [0,1]
            goal_obs = [sample[i] for i in ind_pos]
            goal_obs = np.array(goal_obs)
            humans = []
            for human_num in range(2, sample.size()[0],7):
                humans.append(sample[human_num:human_num + 4])

            return goal_obs, humans

        def compute_distance_from_origin(pos):
            """Compute Euclidean distance between origin (0, 0) and a point."""
            return np.sqrt(pos[0] ** 2 + pos[1] ** 2)



        for i in range(num_episodes):
            o = self.env.reset()
            total_reward = 0
            done = False
            t = 0
            idle_time = 0
            path_length = 0
            personal_space_invasions = 0
            prev_vel = np.array([0.0, 0.0])
            jerk = np.array([0.0, 0.0])
            prev_acc = np.array([0.0, 0.0])
            jerk_count = 0
            velocity_sum = np.array([0.0, 0.0])
            # total_out_of_map = 0
            # total_discomfort_count = 0
            # total_human_collision = 0
            # total_reached_goal = 0
            # total_max_steps = 0
            prev_pos = o["goal"][-2:]
            o = self.preprocess_observation(o)
            while not done:
                t = t + 1

                act_continuous, act_discrete = self.get_action(o, 0)
                new_state, reward, done, info = self.env.step(act_continuous)
                new_state__ = new_state
                new_state = self.preprocess_observation(new_state)
                total_reward += reward

                # self.env.render()

                if info["REACHED_GOAL"]:
                    successive_runs += 1
                    reached_goal_count += 1
                # Update the counters based on the info
                if info["OUT_OF_MAP"]:
                    out_of_map_count += 1
                if info["HUMAN_COLLISION"]:
                    human_collision_count += 1
                # if info["REACHED_GOAL"]:
                #     reached_goal_count += 1
                if info["MAX_STEPS"]:
                    max_steps_count += 1
                if info["DISCOMFORT_CROWDNAV"]:
                    discomfort_count += 1
                # Calculate idle time
                if act_discrete == 3:
                    idle_time += 1


                _, humans_obs = transform_processed_observation_into_raw(torch.from_numpy(new_state))
                # Calculate distance from agent (0, 0) to each human
                for human_pos in humans_obs:
                    if compute_distance_from_origin(human_pos) < personal_space_radius:
                        personal_space_invasions += 1
                        break  

                # Calculate agent velocity, path length and jerk
                current_pos = new_state__["goal"][-2:]
                current_vel = (current_pos - prev_pos) / t
                # print("current_pos")
                # print(current_pos)
                # print("")
                # print("current_vel")
                # print(current_vel)
                # print("")
                # print("prev_pos")
                # print(prev_pos)
                # print("")

                path_length += np.linalg.norm(current_pos - prev_pos)
                current_acc = (current_vel - prev_vel) / t
                jerk = (current_acc - (prev_vel - prev_acc) / t) / t
                o = new_state

                # Calculate number of jerks
                if np.linalg.norm(jerk) > 0.01:  # Threshold for jerk
                    jerk_count += 1

                # personal_space_compliance = (t - personal_space_invasions) / t


                # Sum the velocities for later calculation of average velocity
                velocity_sum += current_vel

                # Update previous values for next calculation
                prev_pos = current_pos
                prev_vel = current_vel
                prev_acc = current_acc

            personal_space_compliance = (t - personal_space_invasions) / t

            # Calculate average velocity
            average_velocity = velocity_sum / t
            # Compute the magnitude of the velocity
            average_velocity = np.linalg.norm(average_velocity)
            
            # Update total values
            # total_jerk_count += (jerk_count/t)
            # total_velocity += average_velocity
            # total_path_length += path_length
            # total_time += t
            # total_idle_time += (idle_time/t)

            # # Print the metrics
            # print(f"Idle Time: {idle_time * t}s")
            # print(f"Path Length: {path_length}")
            # print(f"Final Velocity: {current_vel}")
            # print(f"Final Jerk: {jerk}")
            # total_out_of_map += out_of_map_count  # /num_episodes
            # total_discomfort_count += (discomfort_count   / t)
            # total_human_collision += human_collision_count  # / num_episodes
            # total_reached_goal += reached_goal_count  # / num_episodes
            # total_max_steps += max_steps_count  # / num_episodes

            reward_per_episode += total_reward
            print("Episode [{}/{}] finished after {} timesteps".format(i +
                  1, num_episodes, t), flush=True)

            # Append the values for each episode
            discomfort_counts.append(discomfort_count )
            jerk_counts.append(jerk_count )
            velocities.append(np.linalg.norm(average_velocity))
            path_lengths.append(path_length)
            times.append(t)
            out_of_maps.append(out_of_map_count)
            human_collisions.append(human_collision_count)
            reached_goals.append(reached_goal_count)
            max_steps.append(max_steps_count)
            episode_run.append(i)
            successive_run.append(successive_runs)
            # episode_reward.append(reward_per_episode)
            episode_reward.append(total_reward)
            idle_times.append(idle_time )
            personal_space_compliances.append(personal_space_compliance)

            t = 0
            successive_runs = 0
            out_of_map_count = 0
            human_collision_count = 0
            path_length = 0
            max_steps_count = 0
            discomfort_count = 0   
            reached_goal_count = 0
            # reward_per_episode = 0
            idle_time = 0   
            # jerk_count = 0
            # self.velocity_sum = 0
            
            
        # # Calculate averages over all episodes
        # avg_jerk_count = total_jerk_count / num_episodes
        # avg_velocity = total_velocity / num_episodes
        # avg_path_length = total_path_length / num_episodes
        avg_time = total_time / num_episodes
        # avg_idle_time = total_idle_time / num_episodes

        # # Calculate averages for the counters
        # avg_out_of_map = total_out_of_map  /num_episodes
        # avg_discomfort_count = total_discomfort_count  / num_episodes
        # avg_human_collision = total_human_collision  / num_episodes
        # avg_reached_goal = total_reached_goal  / num_episodes
        # avg_max_steps = total_max_steps  / num_episodes

        # Print the averages
        print(f"Average Idle Time Count: {np.mean(idle_times)}")
        print(f"Average Discomfort Count: {np.mean(discomfort_counts)}")
        print(f"Average Jerk Count: {np.mean(jerk_counts)}")
        print(f"Average Velocity: {np.mean(velocities)}")
        print(f"Average Path Length: {np.mean(path_lengths)}")
        print(f"Average Time: {avg_time}")
        print(f"Average Out of Map: {np.mean(out_of_maps)}")
        print(f"Average Human Collision: {np.mean(human_collisions)}")
        print(f"Average Reached Goal: {np.mean(reached_goals)}")
        print(f"Average Max Steps : {np.mean(max_steps)}")
        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {np.sum(successive_run)}")
        print(f"Average reward per episode: {np.mean(episode_reward)}")
        print(f"Personal Space Compliances: {np.mean(personal_space_compliances)}")

        print(f"Total Idle Time Count: {np.sum(idle_times)}")
        print(f"Total Discomfort Count: {np.sum(discomfort_counts)}")
        print(f"Total Jerk Count: {np.sum(jerk_counts)}")
        print(f"Total Velocity: {np.sum(velocities)}")
        print(f"Total Path Length: {np.sum(path_lengths)}")
        print(f"Total Time: {avg_time}")
        print(f"Total Out of Map: {np.sum(out_of_maps)}")
        print(f"Total Human Collision: {np.sum(human_collisions)}")
        print(f"Total Reached Goal: {np.sum(reached_goals)}")
        print(f"Total Max Steps : {np.sum(max_steps)}")
        print(f"Total episodes run: {num_episodes}")
        print(f"Total successive runs: {np.sum(successive_run)}")
        print(f"Total Personal Space Compliances: {np.sum(personal_space_compliances)}")

        import pandas as pd
        import matplotlib.pyplot as plt
        if not os.path.exists("RESULTS"):
            os.makedirs("RESULTS")

     # Calculate averages and store them in a dictionary
        averages_dict = {
            "Average Idle Time Count": np.mean(idle_times),
            "Average Discomfort Count": np.mean(discomfort_counts),
            "Average Jerk Count": np.mean(jerk_counts),
            "Average Velocity": np.mean(velocities),
            "Average Path Length": np.mean(path_lengths),
            "Average Time": avg_time,
            "Average Out of Map": np.mean(out_of_maps),
            "Average Human Collision": np.mean(human_collisions),
            "Average Reached Goal": np.mean(reached_goals),
            "Average Max Steps": np.mean(max_steps),
            "Total episodes run": num_episodes,
            "Total successive runs": np.sum(successive_run),
            "Average reward per episode": np.mean(episode_reward),
            "Personal Space Compliances": np.mean(personal_space_compliances)

        }
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame([averages_dict])

        # Save the DataFrame to a CSV file
        df.to_csv('RESULTS/duelingDQNAveragesresults.csv', index=False)

        # Save the DataFrame to a JSON file
        df.to_json('RESULTS/duelingDQNAveragesresults.json', orient='records')


        # Create a DataFrame with the collected data
        data = pd.DataFrame({
            'Human Discomfort': discomfort_counts,
            'Jerk Counts': jerk_counts,
            'Velocities': velocities,
            'Distance Traveled': path_lengths,
            'Simulation Time': times,
            'Wall Collisions': out_of_maps,
            'Human Collisions': human_collisions,
            'Reached Goal': reached_goals,
            'Max Steps': max_steps,
            'Episode Run': episode_run,
            'Successful Run': successive_run,
            'Reward': episode_reward,
            'Idle Time': idle_times,
            'Personal Space Compliances Rate': personal_space_compliances

        })



        # Save DataFrame to csv
        data.to_csv('RESULTS/duelingDQNresults.csv', index=False)

        # Save DataFrame to json
        data.to_json('RESULTS/duelingDQNresults.json', orient='records')

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.suptitle('Line Plots for Episode Metrics - DuelingDQN + Predicted Hidden State', fontsize=10)

        # Plot Idle Times
        plt.subplot(3,4,1)
        plt.plot(idle_times)
        plt.title('Idle Times')
        plt.xlabel('Number of Episode')
        plt.ylabel('Idle Time')

        # Plot Successive Run Counts
        plt.subplot(3,4,2)
        plt.plot(successive_run)
        plt.title('Successive Run')
        plt.xlabel('Number of Episode')
        plt.ylabel('Successive Run Count')

        # Plot Episode Reward Counts
        plt.subplot(3,4,3)
        plt.plot(episode_reward)
        plt.title('Episode Reward')
        plt.xlabel('Number of Episode')
        plt.ylabel('Episode Reward')

        # Plot Discomfort Counts
        plt.subplot(3,4,4)
        plt.plot(discomfort_counts)
        plt.title('Discomfort Counts')
        plt.xlabel('Number of Episode')
        plt.ylabel('Discomfort Count')

        # Plot Jerk Counts
        plt.subplot(3,4,5)
        plt.plot(jerk_counts)
        plt.title('Jerk Counts')
        plt.xlabel('Number of Episode')
        plt.ylabel('Jerk Count')

        # Plot Velocities
        plt.subplot(3,4,6)
        plt.plot(velocities)
        plt.title('Velocities')
        plt.xlabel('Number of Episode')
        plt.ylabel('Velocity')

        # Plot Path Lengths
        plt.subplot(3,4,7)
        plt.plot(path_lengths)
        plt.title('Path Lengths')
        plt.xlabel('Number of Episode')
        plt.ylabel('Path Length')

        # Plot Times
        plt.subplot(3,4,8)
        plt.plot(times)
        plt.title('Times')
        plt.xlabel('Number of Episode')
        plt.ylabel('Time')

        # Plot Out of Maps
        plt.subplot(3,4,9)
        plt.plot(out_of_maps)
        plt.title('Out of Maps')
        plt.xlabel('Number of Episode')
        plt.ylabel('Out of Map Count')

        # Plot Human Collisions
        plt.subplot(3,4,10)
        plt.plot(human_collisions)
        plt.title('Human Collisions')
        plt.xlabel('Number of Episode')
        plt.ylabel('Human Collision Count')

        # # Plot Reached Goals
        # plt.subplot(3,4,11)
        # plt.plot(self.reached_goals)
        # plt.title('Reached Goals')
        # plt.xlabel('Number of Episode')
        # plt.ylabel('Reached Goal Count')

        # Plot Personal Space Compliances
        plt.subplot(3,4,11)
        plt.plot(personal_space_compliances)
        plt.title('Personal Space Compliances')
        plt.xlabel('Number of Episode')
        plt.ylabel('Compliance Rate')

        # Plot Max Steps
        plt.subplot(3,4,12)
        plt.plot(max_steps)
        plt.title('Max Steps')
        plt.xlabel('Number of Episode')
        plt.ylabel('Max Step Count')

        plt.tight_layout()
        plt.show()

############################################# HIST ######################################
        fig, axs = plt.subplots(3, 4, figsize=(15, 5))
        plt.suptitle('Line Plots for Episode Metrics - DuelingDQN Model', fontsize=10)

        # Plot Idle Times
        axs[0, 0].hist(idle_times, bins=30)
        axs[0, 0].set_title('Idle Times Histogram')
        axs[0, 0].set_xlabel('Idle Time')
        axs[0, 0].set_ylabel('Frequency')

        # Plot Successive Run Counts
        axs[0, 1].hist(successive_run, bins=30)
        axs[0, 1].set_title('Successive Run Histogram')
        axs[0, 1].set_xlabel('Successive Run Count')
        axs[0, 1].set_ylabel('Frequency')

        # Plot Episode Reward Counts
        axs[0, 2].hist(episode_reward, bins=30)
        axs[0, 2].set_title('Episode Reward Histogram')
        axs[0, 2].set_xlabel('Episode Reward')
        axs[0, 2].set_ylabel('Frequency')

        # Plot Discomfort Counts
        axs[0, 3].hist(discomfort_counts, bins=30)
        axs[0, 3].set_title('Discomfort Counts Histogram')
        axs[0, 3].set_xlabel('Discomfort Count')
        axs[0, 3].set_ylabel('Frequency')

        # Plot Jerk Counts
        axs[1, 0].hist(jerk_counts, bins=30)
        axs[1, 0].set_title('Jerk Counts Histogram')
        axs[1, 0].set_xlabel('Jerk Count')
        axs[1, 0].set_ylabel('Frequency')

        # Plot Velocities
        axs[1, 1].hist(velocities, bins=30)
        axs[1, 1].set_title('Velocities Histogram')
        axs[1, 1].set_xlabel('Velocity')
        axs[1, 1].set_ylabel('Frequency')

        # Plot Path Lengths
        axs[1, 2].hist(path_lengths, bins=30)
        axs[1, 2].set_title('Path Lengths Histogram')
        axs[1, 2].set_xlabel('Path Length')
        axs[1, 2].set_ylabel('Frequency')

        # Plot Times
        axs[1, 3].hist(times, bins=30)
        axs[1, 3].set_title('Times Histogram')
        axs[1, 3].set_xlabel('Time')
        axs[1, 3].set_ylabel('Frequency')

        # Plot Out of Maps
        axs[2, 0].hist(out_of_maps, bins=30)
        axs[2, 0].set_title('Out of Maps Histogram')
        axs[2, 0].set_xlabel('Out of Map Count')
        axs[2, 0].set_ylabel('Frequency')

        # Plot Human Collisions
        axs[2, 1].hist(human_collisions, bins=30)
        axs[2, 1].set_title('Human Collisions Histogram')
        axs[2, 1].set_xlabel('Human Collision Count')
        axs[2, 1].set_ylabel('Frequency')

        # # Plot Reached Goals
        # axs[2, 2].hist(self.reached_goals, bins=30)
        # axs[2, 2].set_title('Reached Goals Histogram')
        # axs[2, 2].set_xlabel('Reached Goal Count')
        # axs[2, 2].set_ylabel('Frequency')

        # Plot Personal Space Compliances Histogram
        axs[2, 2].hist(personal_space_compliances, bins=30)
        axs[2, 2].set_title('Personal Space Compliances Histogram')
        axs[2, 2].set_xlabel('Compliance Rate')
        axs[2, 2].set_ylabel('Frequency')

        # Plot Max Steps
        axs[2, 3].hist(max_steps, bins=30)
        axs[2, 3].set_title('Max Steps Histogram')
        axs[2, 3].set_xlabel('Max Step Count')
        axs[2, 3].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    env = gym.make("SocNavEnv-v1")
    env.configure("./configs/env_timestep_1.yaml")
    env.set_padded_observations(True)
    env.seed(123)

    # config file for the model
    config = "./configs/duelingDQN.yaml"
    # env.observation_space["goal"].shape[0] + env.observation_space["humans"].shape[0] + env.observation_space["laptops"].shape[0] + env.observation_space["tables"].shape[0] + env.observation_space["plants"].shape[0]
    input_layer_size = 23
    agent = DuelingDQNAgent(
        env, config, input_layer_size=input_layer_size, run_name="duelingDQN_SocNavEnv")
    agent.eval(num_episodes=500, path=None)
