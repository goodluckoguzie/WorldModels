import time
import gym
import numpy as np
import torch
import torch.nn as nn
import time
import gym
import numpy as np
import os
import pygame
import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch

from socnavenv.wrappers import WorldFrameObservations

env = gym.make("SocNavEnv-v1")
env.configure("./configs/env_timestep_1.yaml")
env = WorldFrameObservations(env)
env.seed(123)
# env.reset()
# env.render()

from simplejson import load
import os
import pygame
import numpy as np 
import matplotlib.pyplot as plt
pygame.init()
pygame.joystick.init()
controller = pygame.joystick.Joystick(0)


axis_data = { 0:0, 1:0}
button_data = {}
hat_data = {}


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.p1 = (x1, y1)
        self.p2 = (x2, y1)
        self.p3 = (x2, y2)
        self.p4 = (x1, y2)

def get_wall_points(wall):
    assert wall.length != None and wall.thickness != None, "Length or thickness is None type."
    assert wall.x != None and wall.y != None and wall.orientation != None, "Coordinates or orientation are None type"
    
    x1 = wall.x + wall.length / 2 * np.cos(wall.orientation) - wall.thickness / 2 * np.sin(wall.orientation)
    y1 = wall.y + wall.length / 2 * np.sin(wall.orientation) + wall.thickness / 2 * np.cos(wall.orientation)

    x2 = wall.x + wall.length / 2 * np.cos(wall.orientation) + wall.thickness / 2 * np.sin(wall.orientation)
    y2 = wall.y + wall.length / 2 * np.sin(wall.orientation) - wall.thickness / 2 * np.cos(wall.orientation)

    x3 = wall.x - wall.length / 2 * np.cos(wall.orientation) + wall.thickness / 2 * np.sin(wall.orientation)
    y3 = wall.y - wall.length / 2 * np.sin(wall.orientation) - wall.thickness / 2 * np.cos(wall.orientation)

    x4 = wall.x - wall.length / 2 * np.cos(wall.orientation) - wall.thickness / 2 * np.sin(wall.orientation)
    y4 = wall.y - wall.length / 2 * np.sin(wall.orientation) + wall.thickness / 2 * np.cos(wall.orientation)

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def preprocess_human_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )
    # return torch.from_numpy(obs2)
    return obs2
    
def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    observation = np.array([], dtype=np.float32)
    observation = np.concatenate((observation, obs["goal"].flatten()) )
    observation = np.concatenate((observation, obs["humans"].flatten()) )
    observation = np.concatenate((observation, obs["laptops"].flatten()) )
    observation = np.concatenate((observation, obs["tables"].flatten()) )
    observation = np.concatenate((observation, obs["plants"].flatten()) )
    return torch.from_numpy(observation)

def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):
    ind_pos = [6,7]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)

    ind_pos = [8,9,10,11]
    robot_obs = [sample[i] for i in ind_pos]
    robot_obs = np.array(robot_obs)

    humans = []
    for human_num in range(20, sample.size()[0],13):
        humans.append(sample[human_num:human_num + 4])

    return goal_obs,robot_obs, humans



num_episodes = 50
def axis_data_to_action(axis_data):
    return np.array([-axis_data[1], -axis_data[0]])    




##############################################################################
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



def compute_distance(pos1, pos2):
    """Compute Euclidean distance between two points."""
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

##################################################################################

for i in range(num_episodes):
    done = False
    raw_obs = env.reset()

#################################################################################

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

#################################################################################

    # Extract wall information from the environment
    env_walls = []
    for wall in env.walls:
        wall_points = get_wall_points(wall)
        env_walls.append(Wall(*wall_points[0], *wall_points[2]))

    done = False
    t = 0
    goal_obs, robot_obs, humans = transform_processed_observation_into_raw(preprocess_observation(raw_obs))

    robot_pos = np.array([robot_obs[0], robot_obs[1]])
    prev_pos = robot_pos

    while not done:

        t += 1
        env.render()
        goal_obs, robot_obs, humans = transform_processed_observation_into_raw(preprocess_observation(raw_obs))

        # Calculate new velocity using

        robot_pos = np.array([robot_obs[0], robot_obs[1]])
        goal_pos = np.array([goal_obs[0], goal_obs[1]])


        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                axis_data[event.axis] = round(event.value,2)
            elif event.type == pygame.JOYBUTTONDOWN:
                button_data[event.button] = True
            elif event.type == pygame.JOYBUTTONUP:
                button_data[event.button] = False
            elif event.type == pygame.JOYHATMOTION:
                hat_data[event.hat] = event.value

        # Insert your code on what you would like to happen for each event here!
        action = axis_data_to_action(axis_data)

        # print("action", action)
        obs, reward, done, info = env.step(action)
        # print('reward',reward)
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
        # # Calculate idle time
        # if act_discrete == 3:
        #     idle_time += 1

        goal_obs, robot_obs, humans_obs = transform_processed_observation_into_raw(torch.from_numpy(preprocess_human_observation(obs))) 

        if (prev_pos == robot_pos).all():
        # do something
            idle_time += 1

        # Calculate agent velocity, path length and jerk
        current_pos = robot_pos
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

        # Calculate distance from agent to each human
        for human_pos in humans_obs:
            if compute_distance(current_pos, human_pos) < personal_space_radius:
                personal_space_invasions += 1
                break  # No need to check other humans once we found one violation

        path_length += np.linalg.norm(current_pos - prev_pos)
        current_acc = (current_vel - prev_vel) / t
        jerk = (current_acc - (prev_vel - prev_acc) / t) / t

############################################################################

        raw_obs = obs
###############################################################################



        # Calculate number of jerks
        if np.linalg.norm(jerk) > 0.01:  # Threshold for jerk
            jerk_count += 1

        personal_space_compliance = (t - personal_space_invasions) / t

        # Sum the velocities for later calculation of average velocity
        velocity_sum += current_vel

        # Update previous values for next calculation
        prev_pos = current_pos
        prev_vel = current_vel
        prev_acc = current_acc

    # Calculate average velocity
    average_velocity = velocity_sum / t
    # Compute the magnitude of the velocity
    average_velocity = np.linalg.norm(average_velocity)
    
    # Update total values
    # total_jerk_count += (jerk_count/t)
    # total_velocity += average_velocity
    # total_path_length += path_length
    total_time += t
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

    print("Episode [{}/{}] finished after {} timesteps".format(i +
            1, num_episodes, t), flush=True)

    # Append the values for each episode
    discomfort_counts.append(discomfort_count / t)
    jerk_counts.append(jerk_count / t)
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
    idle_times.append(idle_time / t)
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
df.to_csv('RESULTS/JoyStickAveragesresults.csv', index=False)

# Save the DataFrame to a JSON file
df.to_json('RESULTS/JoyStickAveragesresults.json', orient='records')


# Create a DataFrame with the collected data
data = pd.DataFrame({
    'Discomfort Counts': discomfort_counts,
    'Jerk Counts': jerk_counts,
    'Velocities': velocities,
    'Path Lengths': path_lengths,
    'Times': times,
    'Out of Maps': out_of_maps,
    'Human Collisions': human_collisions,
    'Reached Goals': reached_goals,
    'Max Steps': max_steps,
    'Episode Run': episode_run,
    'Ruccessive Run': successive_run,
    'Episode Reward': episode_reward,
    'Idle Times': idle_times,
    'Personal Space Compliances': personal_space_compliances

})

# Save DataFrame to csv
data.to_csv('RESULTS/JoyStickresults.csv', index=False)

# Save DataFrame to json
data.to_json('RESULTS/JoyStickresults.json', orient='records')

# Plot results
plt.figure(figsize=(10, 6))

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
fig, axs = plt.subplots(3, 4, figsize=(10, 10))

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


env.close()