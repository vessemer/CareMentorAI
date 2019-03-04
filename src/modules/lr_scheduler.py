import matplotlib.pyplot as plt
import math


class PiloExt:
    def __init__(
        self, optimizer, 
        multiplier=.1,
        coeff=1.,
        steps_per_epoch=None
    ):

        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_iterations = steps_per_epoch

        self.param_groups_old = list()
        for param_group in self.optimizer.param_groups:
            self.param_groups_old.append(float(param_group['lr']))

        self.iteration = 0
        self.history = {}
        self.coeff = coeff
        self.batch_step(self.iteration)
        
    def get_lr(self):
        '''Calculate the learning rate.'''
        x = float(self.iteration % self.total_iterations) / self.total_iterations

        lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(self.param_groups_old[i] * (1 - x) + self.param_groups_old[i] * self.multiplier * x)
        return lrs

    def batch_step(self, batch_iteration=None, logs=None):
        self.iteration = batch_iteration or self.iteration + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if logs is not None:
            self.history.setdefault('lr', []).append(lr)
            self.history.setdefault('iterations', []).append(self.iteration)

    def step(self, batch_iteration=None, logs=None):
        for i, param_group in enumerate(self.param_groups_old):
            self.param_groups_old[i] *= self.coeff
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['lr'])
#         plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')


class CosinePiloExt(PiloExt):
    def get_lr(self):
        '''Calculate the learning rate.'''
        x = float(self.iteration % self.total_iterations) / self.total_iterations

        lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(
                self.param_groups_old[i] * self.multiplier 
                + (self.param_groups_old[i] - self.param_groups_old[i] * self.multiplier) 
                * (1 + math.cos(math.pi * x)) / 2
            )
        return lrs
