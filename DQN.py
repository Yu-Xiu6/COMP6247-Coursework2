import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
from matplotlib.colors import ListedColormap
from read_maze import load_maze, get_local_maze_information

load_maze()


class Set_Environment:
    def __init__(self, x_size=199, y_size=199, start_x=1, start_y=1, input_size=9, fire=True):
        self.timestep = 0
        self.maze_size = 201
        self.MazeSize_x = x_size
        self.MazeSize_y = y_size
        self.maze = torch.zeros((self.MazeSize_x, self.MazeSize_y, 2))
        self.x = start_x
        self.y = start_y  # Initial position:(1,1)
        self.num_action = 5  # 5 actions: stay,up,down,left,right
        self.input_size = input_size
        self.fire = fire
        # Make the initial observation
        self.around = torch.tensor(get_local_maze_information(self.x, self.y))
        if not self.fire:  # If there is no fire at the initial position
            self.around[:, :, 1] = 0
        self.update_maze()

    def step(self, action): # return all features after doing the action
        reward = self.do_action(action)
        new_x, new_y = self.x, self.y
        observation_ = self.get_observation(new_y, new_x)
        done, done_reward= self.set_game_end()
        done = int(done)
        return observation_, reward, done, new_x, new_y

    def get_observation(self, y, x): # return the state
        state = get_local_maze_information(x, y)
        observation_walls = state[:, :, 0]
        observation_fires = state[:, :, 1]
        modified_walls = observation_walls.flatten()
        modified_fires = observation_fires.flatten()
        new_state = np.array((modified_walls, modified_fires)) # array(2,9), walls and fires state of 9 positions
        return new_state

    def update_maze(self): # update the state to maze
        self.maze[:, :, 1] = torch.where(self.maze[:, :, 1] > 0, self.maze[:, :, 1] - 1.0, self.maze[:, :, 1])
        self.maze[self.y - 1:self.y + 2, self.x - 1:self.x + 2] = self.around

    def check_legal_action(self): # checking the action is legal or not
        legal_actions = []
        # print(self.around[:,:,0])
        # Check left:
        if self.around[1][0][0] == 1 and self.around[1][0][1] == 0 and self.x - 1 >= 0 and self.x - 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            legal_actions.append(1)
        # Check Right:
        if self.around[1][2][0] == 1 and self.around[1][2][1] == 0 and self.x + 1 >= 0 and self.x + 1 < self.maze_size and self.y >= 0 and self.y < self.maze_size:
            legal_actions.append(2)
        # Check Up:
        if self.around[0][1][0] == 1 and self.around[0][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y - 1 >= 0 and self.y - 1 < self.maze_size:
            legal_actions.append(3)
        # Check Down:
        if self.around[2][1][0] == 1 and self.around[2][1][1] == 0 and self.x >= 0 and self.x < self.maze_size and self.y + 1 >= 0 and self.y + 1 < self.maze_size:
            legal_actions.append(4)
        return legal_actions

    def get_next_position(self, action): # get the next position after actions
        if action == 0:  # Stay
            x = self.x
            y = self.y
        elif action == 1:  # Left
            x = self.x - 1
            y = self.y
        elif action == 2:  # Right
            x = self.x + 1
            y = self.y
        elif action == 3:  # Up
            x = self.x
            y = self.y - 1
        elif action == 4:  # Down
            x = self.x
            y = self.y + 1
        else:
            # print(action)
            raise ValueError(f"Unknown Action: {action}")
        return x, y

    def do_action(self, action): # do the action
        reward = -1.0
        if action not in self.check_legal_action():  # If action is illegal, stay at current position and discount reward by 1
            action = 0
            reward -= 1
        # Get the next position after this action
        self.x, self.y = self.get_next_position(action)
        # Update agent states
        self.timestep += 1
        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()

        done, done_reward = self.set_game_end()
        if done:
            return done_reward+reward
        return reward

    def set_game_end(self): # game over: 1.reach the destination 2.fire on the current position
        if self.x == self.MazeSize_x and self.y == self.MazeSize_y:  # 201-1-wall
            reward = 100
            return True, reward
        if self.around[1, 1, 1] > 0:  # fire on the current position
            reward = -100
            return True, reward
        return False, 0

    def restart_game(self, x=1, y=1): # restart the game
        # Initial all the situation to the beginning
        self.timestep = 0
        self.x = x
        self.y = y

        self.around = torch.tensor(get_local_maze_information(self.y, self.x))
        if not self.fire:
            self.around[:, :, 1] = 0
        self.update_maze()


np.random.seed(1)
tf.set_random_seed(1)


class DQN:
    def __init__(
            self,
            env,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.env = env
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0  # total learning step
        self.memory = np.zeros((self.memory_size, 40))  # initialize zero memory[state,action,reward,state_]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.InteractiveSession()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        tf.reset_default_graph()
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                # print(self.s)
                # print(w1)
                # print(b1)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

                # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def save_maze_policy(self, path):
        """Save the current policy as an image"""
        policy_image = torch.zeros((self.env.maze_size, self.env.maze_size))
        for y in range(self.env.maze_size):
            for x in range(self.env.maze_size):
                if self.env.maze[y, x, 0]:  # not a wall
                    # Comput the best action
                    state = self.get_state(y, x).to(self.device)
                    action_values = self.q_net(self.feature_net(state.unsqueeze(0)))[0]
                    best_action = torch.argmax(action_values)
                    policy_image[y, x] = best_action

        # Plot heatmap to show frequency of visiting each position
        newcmap = ListedColormap(['black', 'blue', 'red', 'yellow', 'white'])
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(policy_image, cmap=newcmap, vmin=0, vmax=4)
        fig.colorbar(im)
        plt.savefig(path)
        plt.close(fig)

    def save_visit_frequency(self, path):
        # Save a heatmap showing frequency of visiting each position
        fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(self.visited_times, cmap='gray')
        fig.colorbar(im)
        plt.savefig(path)
        plt.close(fig)

    def save_explored_maze(self, path):
        """Save the explored positions as an image"""
        plt.figure(figsize=(20, 20))
        plt.imshow((self.visited_times > 0), cmap="gray")
        plt.savefig(path)
        plt.close()

    def store_transition(self, s, a, r, s_): # store the memory
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        total = np.array([a, r])
        total = np.expand_dims(total, 0).repeat(2, axis=0)

        transition = np.hstack((s, total, s_))
        index = self.memory_counter % self.memory_size
        # print(index)
        transition = transition.flatten()
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation.reshape(9, 2)
        while True: # choose action based on Q value
            if np.random.uniform() < self.epsilon:  # choosing action
                actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
                action = np.argmax(actions_value)  # if existing best, choose best action
            else: # choose action based on random
                action = np.random.randint(0, self.n_actions)
            legal_actions = self.env.check_legal_action()

            # print('Legal_action',legal_actions)
            if action in legal_actions: # make sure the action is legal or not
                # print('Raw:',action)
                return action
                break
            elif len(legal_actions) == 0: # if no action can be chosen, the agent will stay there.
                action = 0
                return action

    def learn(self):
        tf.initialize_all_variables().run()
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:]})  # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


def train(RL, env):
    step = 0
    train_x, train_y = 1, 1
    for episode in range(500):
        # initial observation

        env.restart_game()
        while True:
            # agent take action based on observation
            print('Current position:({},{})'.format(train_x, train_y))
            observation = env.get_observation(train_y, train_x)

            action = RL.choose_action(observation)
            # print('action',action)
            # agent take action and get next observation and reward
            observation_, reward, done, new_x, new_y = env.step(action)
            train_x, train_y = new_x, new_y
            print('New position:({},{})|Chosen action:{}'.format(new_x, new_y, action))
            # print('Done',done)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 500) and (step % 10 == 0):
                RL.learn()

            # observation = observation_

            if done:
                break
            step += 1
        if episode % 50 == 0:
            print('Completed episode {}'.format(episode))
    print('Game Over')

if __name__ == "__main__":
    MEMORY_SIZE = 5000
    ACTION_SPACE = 5
    env = Set_Environment(199, 199, fire=False)
    DQN_network = DQN(
        env, n_actions=ACTION_SPACE, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, learning_rate=0.01, reward_decay=0.9, output_graph=True)

    train(DQN_network, env)