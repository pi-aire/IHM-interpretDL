from random import random, randint
from numpy.testing._private.utils import requires_memory
import torch

from model import DQN
from replayMemory import ReplayMemory
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    def __init__(self, args, n_actions):
        self.model = DQN(args.img_width, args.img_height, args.channels, n_actions).to(device)
        self.n_action = n_actions
        self.epsilon_start = args.epsilon
        self.epsilon = args.epsilon
        self.decay_start = args.decay_start
        self.decay_end = args.n_epochs * 0.8
        self.memory = ReplayMemory(args)
        self.batch_size = args.batch_size
        self.actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def update_esplison(self, epoch):
        if epoch < self.decay_end:
            ep_step = (self.epsilon_start - 0.1) / self.decay_end

            self.epsilon = self.epsilon_start - (ep_step * epoch)
        else:
            self.epsilon = 0

    def act(self, state):
        if random() <= self.epsilon:
            return randint(0, self.n_action - 1)
        else:
            return self.get_best_action(state)

    def get_best_action(self, state):
        q = self.get_q_values(state)
        m, index = torch.max(q, 1)
        action = index.item()

        return action

    def get_q_values(self, state):
        state = torch.from_numpy(state).to(device)
        return self.model(state)

    def get_best_action_wGrad(self, state):
        state = torch.from_numpy(state).to(device)
        state.requires_grad_(True)
        output = self.model(state)
        
        m, index = torch.max(output, 1)
        action = index.item()
        # print(output)
        one_hot = torch.FloatTensor(1,3).zero_().to(device)
        # print(one_hot.size())
        one_hot[0][action] = 1
       
        output.backward(gradient=one_hot)
        
        grads = state.grad.clone()
        grads.squeeze_(0) # le _ c'est pour dire in place opti de calcul

        # print(grads.size())
        
        grads.transpose_(0,1) # On changes l'ordre des dimensions ici on inverse la dim 0 à la dim 1
        grads.transpose_(1,2) # On changes l'ordre des dimensions ici on inverse la dim 1 à la dim 2
        
        grads = np.amax(grads.numpy(), axis=2)
        
        # On suprimme les valeurs négatives
        grads[grads < 0] = 0
        
        # On normalise les valeurs
        grads -= grads.min()
        grads /= grads.max()
        
        grads *= 254
        grads = grads.astype(np.int8)

        return action, grads.transpose((1, 0))
