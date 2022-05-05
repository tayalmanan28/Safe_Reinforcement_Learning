import random
import numpy as np

class ReplayMemory:

    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask, t=None, next_t=None):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, mask, t, next_t)
        self.position = (self.position + 1) % self.capacity

    def batch_push(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, t_batch=None, next_t_batch=None):

        for i in range(state_batch.shape[0]):  # TODO: Optimize This
            if t_batch is not None and next_t_batch is not None:
                self.push(state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i], t_batch[i], next_t_batch[i])  # Append transition to memory
            else:
                self.push(state_batch[i], action_batch[i], reward_batch[i], next_state_batch[i], mask_batch[i])  # Append transition to memory

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask, t, next_t = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask, t, next_t

    def __len__(self):
        return len(self.buffer)
