import causalchamber.datasets as CCDatasets
from causalchamber.models import model_f3
import causalchamber
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import plotly
from torchvision import transforms
from PIL import Image
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def compute_malus_factor(theta_1, theta_2):
    # Transform parameters
    theta_1, theta_2 = torch.deg2rad(theta_1), torch.deg2rad(theta_2)
    malus_factor = torch.cos(theta_1 - theta_2) ** 2
    return malus_factor

def draw_image_from_tensor(tensor_sim, tensor_real):
    
    image_sim = tensor_sim.cpu().detach().numpy()
    image_real = tensor_real.cpu().detach().numpy()


    # Convert the PIL images to numpy arrays for plotting
    image_sim_np = np.round(image_sim * 255).astype(np.uint8)
    image_real_np = image_real.astype(np.uint8)

    # Create a subplot with 1 row and 2 columns
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Simulated Image', 'Real Image'))

    # Add the simulated image to the first subplot
    fig.add_trace(
        go.Image(z=image_sim_np),
        row=1, col=1
    )

    # Add the real image to the second subplot
    fig.add_trace(
        go.Image(z=image_real_np),
        row=1, col=2
    )

    # Update layout
    fig.update_layout(height=600, width=1000, title_text="Simulated vs Real Image")

    # Show the figure
    pio.show(fig)


def read_image(image_path):
    image = Image.open(image_path)
    image_tensor = torch.as_tensor(np.array(image).astype('float'))
    return image_tensor
# For light tunnel: the params \theta are the RGB values and the dimming level defined by the polarizers angles (\theta:= [R,G,B, \alpha])

class LightTunnel(Dataset):
    def __init__(self, exp_name, idx_offset, num_samples, is_label=False, is_noisy_label=False):
        self.data_path = os.path.join(os.environ.get('VIRTUAL_HOME'), 'data/light_tunnel')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            dataset = CCDatasets.Dataset('lt_camera_v1', root=self.data_path, download=True)
        else:
            dataset = CCDatasets.Dataset('lt_camera_v1', root=self.data_path, download=False)
        experiment = dataset.get_experiment(name=exp_name)
        self.df = experiment.as_pandas_dataframe()
        self.experiment = exp_name
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.df = self.df.iloc[idx_offset:idx_offset+num_samples]
        self.is_label = is_label
        self.is_noisy_label = is_noisy_label

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pol_1 = torch.tensor(self.df['pol_1'].iloc[idx])
        pol_2 = torch.tensor(self.df['pol_2'].iloc[idx])
        pol_diff = pol_1-pol_2
        r = torch.tensor(self.df['red'].iloc[idx])
        g = torch.tensor(self.df['green'].iloc[idx])
        b = torch.tensor(self.df['blue'].iloc[idx])
        image_path = os.path.join(self.data_path, 'lt_camera_v1', self.experiment, 
                                            'images_100', self.df['image_file'].iloc[idx])
        image = read_image(image_path)
        image /= 255
        if self.is_label:
            if self.is_noisy_label:
                noise_frac_rgb = self.is_noisy_label   
                noise_frac_pol = self.is_noisy_label    

                # Create a generator seeded uniquely for each idx.
                gen = torch.Generator()
                gen.manual_seed(int(idx))


                # Add noise scaled by the label scales.
                r = r + (255 * noise_frac_rgb * torch.randn(1, generator=gen).item())
                g = g + (255 * noise_frac_rgb * torch.randn(1, generator=gen).item())
                b = b + (255 * noise_frac_rgb * torch.randn(1, generator=gen).item())
                pol_1 = pol_1 + (180 * noise_frac_pol * torch.randn(1, generator=gen).item())
                pol_2 = pol_2 + (180 * noise_frac_pol * torch.randn(1, generator=gen).item())

                # Clamp values to remain in valid ranges.
                r = torch.round(torch.clamp(r, 0, 255))
                g = torch.round(torch.clamp(g, 0, 255))
                b = torch.round(torch.clamp(b, 0, 255))
                pol_1 = torch.round(torch.clamp(pol_1, -180, 180))
                pol_2 = torch.round(torch.clamp(pol_2, -180, 180))

            alpha = compute_malus_factor(pol_1, pol_2)
            return image, (r, g, b, pol_1, pol_2, pol_diff, alpha)
        else:
            return image
    
    def set_normalisers(self, normalisers=None):
        if normalisers is None:
            normalisers = {}
            images = []
            for idx in range(len(self.df)):
                image_path = os.path.join(self.data_path, 'lt_camera_v1', self.experiment, 
                                            'images_100', self.df['image_file'].iloc[idx])
                image = read_image(image_path)
                images.append(image)
            images = torch.stack(images)
            normalisers = {"mean": images.mean(), "std": images.std()}
        self.normalisers = normalisers




class LightTunnelModelF3(Dataset):
    def __init__(self, num_samples=1000, seed_offset=0):
        # Load camera spectral sensitivity
        sens = pd.read_csv("data/resources/camera_sensitivity.csv", index_col='channel')

        # Light-source wavelengths
        color_wavelengths = pd.read_csv("data/resources/light_source_wavelengths.csv", index_col=0)
        wavelengths = color_wavelengths.typical

        # Build S matrix
        S = np.zeros((3,3))
        channels = ['red', 'green', 'blue']
        for i,channel_response in enumerate([sens.loc[c] for c in channels]):
            for j,wavelength in enumerate(wavelengths):
                S[i,j] = np.interp(wavelength, sens.columns.astype(int), channel_response)

        self.S = S
        self.wb = np.array([2.65625, 1.0, 1.77344])
        self.out_params = {"center_x": 0.5, "center_y": 0.5, "radius": 0.22, "offset": 0, "image_size":100}
        self.exposure = 2.5
        self.num_samples = num_samples
        self.Tp = np.array([[0.29, 0.35, 0.33]]).T
        self.Tc = np.array([[0.02, 0.08, 0.18]]).T
        self.seed_offset = seed_offset
    
    def __len__(self):
        return self.num_samples
    
    def simulate_(self, r, g, b, pol_1, pol_2):
        image = model_f3(r, g, b, pol_1, pol_2, **self.out_params, S=self.S, w_r=self.wb[0], w_g=self.wb[1], 
                w_b=self.wb[2], exposure=self.exposure, Tp=self.Tp, Tc=self.Tc)
        return image
    
    '''def to_tensor_(self, array):
        tensor = torch.tensor(array).float().to(self.device)
        return tensor'''

    def __getitem__(self, idx):
        np.random.seed(idx+self.seed_offset) 
        r = np.random.uniform(0, 255)
        g = np.random.uniform(0, 255)
        b = np.random.uniform(0, 255)
        pol_1 = np.random.uniform(-180, 180)
        pol_2 = np.random.uniform(-180, 180)
        pol_diff = pol_1-pol_2
        alpha = compute_malus_factor(torch.tensor(pol_1), torch.tensor(pol_2))
        # Transform parameters
        image = self.simulate_(r, g, b, pol_1, pol_2)
        return torch.tensor(image).float(), (torch.tensor(r), torch.tensor(g), torch.tensor(b), torch.tensor(pol_1), 
        torch.tensor(pol_2), torch.tensor(pol_diff), alpha)
    
    def set_normalisers(self, normalisers=None):
        if normalisers is None:
            normalisers = {}
            images = []
            for idx in range(self.num_samples):
                np.random.seed(idx+self.seed_offset) 
                r = np.random.uniform(0, 255)
                g = np.random.uniform(0, 255)
                b = np.random.uniform(0, 255)
                pol_1 = np.random.uniform(-180, 180)
                pol_2 = np.random.uniform(-180, 180)
                image = self.simulate_(r, g, b, pol_1, pol_2)
                image = torch.tensor(image).float()
                images.append(image)
            images = torch.stack(images)
            normalisers = {"mean": images.mean(), "std": images.std()}

        self.normalisers = normalisers



'''LTdata = LightTunnel('uniform_ap_1.8_iso_500.0_ss_0.005', 0, 1000)
LTdataF3sim = LightTunnelModelF3(1000, 0)
batch = LTdata[0]
batch_sim = LTdataF3sim[0]
draw_image_from_tensor(batch_sim[0], batch[0])
print('done')'''