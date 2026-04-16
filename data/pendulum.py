import numpy as np
import torch
from torch.utils.data import Dataset


class DampedPendulumDataset(Dataset):
    def __init__(self,length, noise_level, idx_offset=0, num_samples=1000, is_label = False, subsample = 10, is_noisy_label=False):
        """
        Args:
        length (int): Number of time steps to simulate.
        noise_level (float): Standard deviation of the noise.
        num_samples (int): Number of samples to generate.
        """
        self.length = length
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.idx_offset = idx_offset
        self.is_label = is_label
        self.sample_phase = True
        self.subsample = subsample
        self.is_noisy_label = is_noisy_label

    def __len__(self):
        return self.num_samples
    
    def set_phase_sample(self, sample_phase):
        self.sample_phase = sample_phase

    def sample_params(self, idx):
        np.random.seed(self.idx_offset+idx)
        # Initialize
        #omega = np.random.uniform(6, 12)  # random frequency
        omega = np.random.uniform(0.1, 1.0) *np.pi
        alpha = np.random.uniform(0.5, 1)
        if self.sample_phase:
            #theta = np.random.uniform(-np.pi, np.pi) # random initial angle
            #theta = np.random.choice([-1, 1]) * np.random.uniform(0.3*np.pi, np.pi)
            theta = np.random.uniform(-np.pi, np.pi) # random initial angle
        else:
            theta = 0.65*-np.pi
        theta_prime = np.random.uniform(-1, 1)  # random initial angular velocity
        return omega, alpha, theta, theta_prime


    def simulate_(self, omega, alpha, theta, theta_prime):
        dt = 0.01  # time step
        data = []

        for _ in range(self.length):
            # Update equations using Euler's method
            d_theta_prime = -alpha * theta_prime - omega * np.sin(theta)
            theta += theta_prime * dt
            theta_prime += d_theta_prime * dt
            data.append(theta + np.random.normal(0, self.noise_level, size=theta.shape))
        
        #subsample
        data = data[::self.subsample]
        return data

    def __getitem__(self, idx):
        
        omega, alpha, theta, theta_prime = self.sample_params(idx)

        data = self.simulate_(omega, alpha, theta, theta_prime)

        if self.is_label:
            if self.is_noisy_label:
                noise_frac = self.is_noisy_label
                # Create a generator seeded uniquely for each idx.
                gen = torch.Generator()
                gen.manual_seed(int(idx))
                # Generate separate noise values for each parameter.
                noise_omega = torch.randn(1, generator=gen).item()
                noise_theta = torch.randn(1, generator=gen).item()
                # Add noise scaled by the label scales.
                omega = omega + (np.pi * noise_frac * noise_omega)
                theta = theta + (np.pi * noise_frac * noise_theta)
            return torch.tensor(data).float(), (torch.tensor(omega).float(), torch.tensor(alpha).float(), 
                    torch.tensor(theta).float(), torch.tensor(theta_prime).float())
        else:
            return torch.tensor(data).float()

class DampedPendulumDependentDataset(DampedPendulumDataset):
    def sample_params(self, idx):
        np.random.seed(self.idx_offset+idx)

        alpha = np.random.uniform(0.05, 0.5)
        omega = (np.pi/2)* alpha + np.random.uniform(0.05, 0.5)*(np.pi/2)
        
        if self.sample_phase:
            #theta = np.random.uniform(-np.pi, np.pi) # random initial angle
            theta = np.random.choice([-1, 1]) * np.random.uniform(0.3*np.pi, np.pi)
        else:
            theta = 0.65*-np.pi
        theta_prime = np.random.uniform(-1, 1)  # random initial angular velocity
        return omega, alpha, theta, theta_prime


class LinearDampedPendulumDatasetClosedForm(Dataset):
    def __init__(self, length, noise_level, idx_offset=0, num_samples=1000, dt=0.01, subsample = 10):
        """
        Args:
        length (int): Number of time steps to simulate.
        noise_level (float): Standard deviation of the noise.
        seed_offset (int): Offset for the random seed.
        num_samples (int): Number of samples to generate.
        dt (float): Time step for the numerical solver.
        """
        self.length = length
        self.noise_level = noise_level
        self.idx_offset = idx_offset
        self.num_samples = num_samples
        self.dt = dt
        self.sample_phase = True
        self.subsample = subsample

    def __len__(self):
        return self.num_samples
    
    def set_phase_sample(self, sample_phase):
        self.sample_phase = sample_phase

    def simulate_(self, omega, alpha, theta, theta_prime):
        data = []

        for _ in range(self.length):
            # Update equations using Euler's method
            d_theta_prime = -alpha * theta_prime - omega * theta
            theta += theta_prime * self.dt
            theta_prime += d_theta_prime * self.dt
            data.append(theta + np.random.normal(0, self.noise_level))
        
        #subsample
        data = data[::self.subsample]
        return data

    def __getitem__(self, idx):
        np.random.seed(self.idx_offset+idx)
        # Initialize
        omega = np.random.uniform(0.1, 1.0) *np.pi
        alpha = np.random.uniform(0.05, 0.5)  # random damping factor
        if self.sample_phase:
            theta = np.random.uniform(-np.pi, np.pi) # random initial angle
        else:
            theta = 0.65*-np.pi
        theta_prime = np.random.uniform(-1, 1)  # random initial angular velocity

        data = self.simulate_(omega, alpha, theta, theta_prime)


        return torch.tensor(data).float(), (torch.tensor(omega).float(), torch.tensor(alpha).float(), 
                    torch.tensor(theta).float(), torch.tensor(theta_prime).float())

class UndampedPendulumDataset(LinearDampedPendulumDatasetClosedForm):
    def simulate_(self, omega, alpha, theta, theta_prime):
        data = []

        for _ in range(self.length):
            # Update equations using Euler's method
            d_theta_prime = -omega * np.sin(theta)
            theta += theta_prime * self.dt
            theta_prime += d_theta_prime * self.dt
            data.append(theta + np.random.normal(0, self.noise_level))
        
        #subsample
        data = data[::self.subsample]
        return data


class UnpairedXdomainDataset(Dataset):
    def __init__(self, xo_dataset, xs_dataset):
        self.xo_dataset = xo_dataset
        self.xs_dataset = xs_dataset
        self.length = min(len(xo_dataset), len(xs_dataset))  # To avoid out of bounds issues
        self.len_xo = len(xo_dataset)
        self.len_xs = len(xs_dataset)
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.len_xo <= self.len_xs:
            xo = self.xo_dataset[index]
            #select random sample from xs
            xs = self.xs_dataset[np.random.randint(0, self.len_xs)]
        else:
            xo = self.xo_dataset[np.random.randint(0, self.len_xo)]
            xs = self.xs_dataset[index]
        return xo, xs
