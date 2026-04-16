import causalchamber.datasets as CCDatasets
from causalchamber.models import model_a1, model_a2, simulator_a1_c2, simulator_a1_c3, simulator_a2_c3
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import plotly
# For wind tunnel: the param \theta is the hutch position H


def draw_signal(signal):
    fig = plotly.graph_objs.Figure()
    fig.add_trace(plotly.graph_objs.Scatter(x=np.arange(len(signal)), y=signal))
    fig.show()


class WindTunnel(Dataset):
    def __init__(self, exp_name, idx_offset, num_samples, is_label=False, is_noisy_label=False):
        self.data_path = os.path.join(os.environ.get('VIRTUAL_HOME'), 'data/wind_tunnel')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            dataset = CCDatasets.Dataset('wt_intake_impulse_v1', root=self.data_path, download=True)
        else:
            dataset = CCDatasets.Dataset('wt_intake_impulse_v1', root=self.data_path, download=False)
        experiment = dataset.get_experiment(name=exp_name)
        df_raw = experiment.as_pandas_dataframe()
        self.df = self.extract_impulses_(df_raw)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
        self.df = self.df.iloc[idx_offset:idx_offset+num_samples]
        self.load_in = df_raw['load_in'].iloc[:50].values
        self.load_out = df_raw['load_out'].iloc[:50].values
        self.timestamps = df_raw['timestamp'].iloc[:50].values- df_raw['timestamp'].iloc[0]
        self.p_amb = np.mean(df_raw['pressure_ambient'].iloc[:50].values)
        self.is_label = is_label
        self.is_noisy_label = is_noisy_label

    # To extract impulses into a new dataframe with one row per impulse
    def extract_impulses_(self, df_raw, field="pressure_downwind"):
        len_impulse = len(df_raw[df_raw.flag == 0])
        n_impulses = len(pd.unique(df_raw.flag))
        impulses = df_raw[field].values.reshape(n_impulses,len_impulse)
        p_amb = df_raw["pressure_ambient"].values.reshape(n_impulses,len_impulse)
        impulses = impulses - p_amb
        params = df_raw.loc[0::50][['hatch', 'load_out', 'osr_downwind']].values
        array = np.hstack([params, impulses])
        return pd.DataFrame(array, columns=["hatch", "load_out", "osr_downwind"] + ["t_%d" % i for i in range(len_impulse)])

    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        signal = torch.tensor(self.df.iloc[idx].values[3:]).float()
        hatch = torch.tensor(self.df.iloc[idx].values[0]).float()
        if self.is_label:
            if self.is_noisy_label:
                gen = torch.Generator()
                gen.manual_seed(int(idx))
                noise_frac = self.is_noisy_label
                # Add noise scaled by the label scales.
                hatch = hatch + (45 * noise_frac * torch.randn(1, generator=gen).item())
            return signal, (hatch, )
        else:
            return signal

class WindTunnelModelA2C3(Dataset):
    def __init__(self, resources, num_samples=1000, idx_offset=0):
        #from notebook in https://github.com/juangamella/causal-chamber-paper/blob/main/case_studies/mechanistic_models.ipynb
        params = np.load(os.path.join('./data/resources', resources), allow_pickle=True).item()
        C_MIN = 0.166
        C_MAX = 0.27
        L_MIN = 0.1
        T = 0.05
        self.args = {
            'omega_max': 3000 * np.pi / 30,
            'I': 0.5 * 0.059**2 * 0.02,
            'Q_max': 186.7 / 3600,
            'S_max': 74.82473949999999,
            'beta': 0.15,
            'r_0': 0.75,
            'barometer_error': 0, 
            'barometer_precision': 0.2
        }
        def tau(L, C_min=C_MIN, C_max=C_MAX, L_min=L_MIN, T=T):
            L = np.atleast_1d(L)
            torques = T * (C_min + np.maximum(L_min, L) ** 3 * (C_max - C_min) - C_min)
            torques[L == 0] = 0
            return torques if len(L) > 1 else torques[0]
        C = tau(1) / self.args['omega_max']**2
        self.args.update({'C': C, 'tau': tau})
        self.args.update(params)
        self.args.update({'omega_in_0': model_a1(params['load_in'][0], L_MIN, self.args['omega_max']), 
                        'omega_out_0': model_a1(params['load_out'][0], L_MIN, self.args['omega_max'])})
        self.num_samples = num_samples
        self.seed_offset = idx_offset
    
    def __len__(self):
        return self.num_samples

    def simulate_(self, hatch):
        self.args.update({'hatch': hatch})
        sim = simulator_a2_c3(**self.args)
        return sim[0]-self.args['P_amb']
    
    def __getitem__(self, idx):
        np.random.seed(idx+self.seed_offset)
        hatch = np.random.uniform(0.1, 45)
        signal = self.simulate_(hatch)
        return torch.tensor(signal).float(), (torch.tensor(hatch).float(), )


'''WTdata = WindTunnel('load_out_0.5_osr_downwind_4', torch.device('cuda'))
real_params = {'load_in': WTdata.load_in, 
'load_out': WTdata.load_out,
'timestamps': WTdata.timestamps,
'P_amb': WTdata.p_amb}

WTsimdata = WindTunnelModelA2C3(real_params, 1000, torch.device('cuda'))
real_signal = WTdata[0][0]
signal = WTsimdata[0][0]

draw_signal(real_signal.cpu().numpy())
draw_signal(signal.cpu().numpy())'''