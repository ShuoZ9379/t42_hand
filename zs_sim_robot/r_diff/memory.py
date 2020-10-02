import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.r_diff = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs_batch = self.observations.get_batch(batch_idxs)
        r_diff_batch = self.r_diff.get_batch(batch_idxs)
        action_batch = self.actions.get_batch(batch_idxs)

        result = {
            'ob': obs_batch,
            'r_diff': r_diff_batch,
            'ac': action_batch,
        }
        return result

    def append(self, obs, action, r_diff_label):        
        self.observations.append(obs)
        self.actions.append(action)
        self.r_diff.append(r_diff_label)

    def data(self, size):
        return self.observations.data[:size], self.actions.data[:size], self.r_diff.data[:size]

    @property
    def nb_entries(self):
        return len(self.observations)
