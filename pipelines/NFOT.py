from .pipeline import BasePipeline, BaseStage
from .backbones import *
from torch import nn
import torch
from .cond_NF import ConditionalMAF
from utils import *
from data import *
import ot
from torch.utils.data import TensorDataset, DataLoader
import wandb
from concurrent.futures import ThreadPoolExecutor

class RoPE_NPE(BaseStage):
    '''Neural Posterior Estimator, trained on the source (simulated) domain'''
    def __init__(self, backbone, params_infer, posterior_train_sample = 100, posterior_test_sample=1000, fold=None):
        super(RoPE_NPE, self).__init__()
        self.params_infer = params_infer
        num_param = len(params_infer) 
        self.label_idx = [val for key, val in params_infer.items()]
        self.encoder = globals()[backbone['name']](**backbone['params']) 
        self.output_size = backbone['params']['output_size']
        self.predictor = ConditionalMAF(input_dim=num_param, context_dim=self.output_size, dropout_prob=0.1)
        self.fold = fold
        self.mse_loss = nn.MSELoss()
        self.posterior_train_sample = posterior_train_sample
        self.posterior_test_sample = posterior_test_sample


    def data_preprocess(self, inputs):
        inputs, labels = inputs
        labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
        return inputs, labels
    
    def metrics_calculate(self,data):
        '''data: tuple of (preds, labels)
        returns a dictionarsy of metrics'''
        preds, labels = data
        scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
        scaled_labels = torch.stack(scaled_labels, dim=1)
        preds, z = preds
        nll_loss = -self.predictor(scaled_labels, context=z)
        l2_reg = self.predictor.regularization_loss()
        loss = torch.mean(nll_loss)+l2_reg
        metrics = {'loss':loss, 'nll_loss':torch.mean(nll_loss), 'l2_reg':l2_reg}
        for ind, name in enumerate(self.params_infer):
            MSELossAbs = self.mse_loss(self.scale_params(preds[:, ind], name, inverse=True), labels[:, ind])
            metrics[f'MSE_{name}'] = MSELossAbs
       
        return metrics

    
    def forward(self, inputs, phase):
        if phase == 'train':
            posterior_sample = self.posterior_train_sample
            self.encoder.train()
            self.predictor.train()
        else:
            posterior_sample = self.posterior_test_sample
            self.encoder.eval()
            self.predictor.eval()
        z = self.encoder(inputs) # context (the condition)
        preds = self.predictor.sample(posterior_sample, z).mean(dim=1)
        return preds, z

    def train_step(self, inputs, labels):
        inputs, labels = self.data_preprocess((inputs, labels))
        preds = self.forward(inputs, 'train')
        metrics = self.metrics_calculate((preds, labels))
        return preds, metrics

    def eval_step(self, inputs, labels):
        inputs, labels = self.data_preprocess((inputs, labels))
        with torch.no_grad():
            preds = self.forward(inputs, 'eval')
            metrics = self.metrics_calculate((preds, labels))
        return preds, metrics
    
    def optimise(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def get_embedding(self, inputs):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(inputs)

class NSE_finetune(RoPE_NPE):
    def __init__(self, backbone, params_infer,  posterior_train_sample, posterior_test_sample, fold=None):
        super(NSE_finetune, self).__init__(backbone, params_infer,  posterior_train_sample, posterior_test_sample, fold)
        self.simulate_ = None
        self.pretrained_encoder = None
        self.params_sim = backbone['params_sim']
    
    def data_preprocess(self, inputs):
        inputs, labels = inputs
        target_preds_gt = self.compute_zs_gt(labels)
        return inputs, target_preds_gt 

    def compute_zs_gt(self, params):
        self.pretrained_encoder.eval()
        device = next(self.pretrained_encoder.parameters()).device
        with torch.no_grad():
            gt_xs_batch = []
            for i in range(params[0].size(0)):
                sample = tuple(p[i].item() for p in params)[:self.params_sim]
                gt_xs_batch.append(torch.tensor(self.simulate_(*sample)).float().to(device).unsqueeze(0))
            gt_xs_batch = torch.cat(gt_xs_batch, dim=0)
            target_preds_gt = self.pretrained_encoder(gt_xs_batch)
        return target_preds_gt.detach()

    def metrics_calculate(self, data):
        preds, labels = data
        preds, z = preds
        metrics = {}
        loss = self.mse_loss(z, labels)       
        metrics['loss'] = loss
        return metrics

class OT_align(BaseStage):
    def __init__(self, params_infer, args, posterior_train_sample=100, posterior_test_sample=1000):
        super(OT_align, self).__init__()
        self.params_infer = params_infer
        self.param_names = list(params_infer.keys())
        self.posterior_train_sample = posterior_train_sample
        self.posterior_test_sample = posterior_test_sample
        self.NPE = None
        self.NSE_ft = None
        self.label_idx = [val for key, val in params_infer.items()]
        self.epsilon = args['gamma']
        self.tau_a = args['tau_a']
        self.tau_b = args['tau_b']
    
    def compute_transport_matrix(self, g_xo, h_xs):
        transport_matrix, _ = compute_semi_balanced_ot(g_xo.detach().cpu().numpy(), h_xs.detach().cpu().numpy(), self.epsilon, self.tau_a, self.tau_b)
        transport_matrix = torch.tensor(transport_matrix.tolist(), dtype=torch.float32).to(g_xo.device)
        return transport_matrix
    
    @staticmethod
    def _worker(arg_tuple):
        test_vec_np, g_xo_OT_np, h_xs_np, epsilon, tau_a, tau_b = arg_tuple
        real_np = np.vstack([test_vec_np, g_xo_OT_np])
        T, _ = compute_semi_balanced_ot(
            real_np, h_xs_np,
            epsilon=epsilon, tau_a=tau_a, tau_b=tau_b
        )
        return T[-1, :]
    
    def compute_single_sample_OT(self,
    g_xo_test: torch.Tensor,
    g_xo_OT:   torch.Tensor,
    h_xs:      torch.Tensor,
    max_workers: int = None
    ) -> torch.Tensor:
        """
        Parallel‐CPU version of your original implementation using POT.

        Args:
        g_xo_test: [B, D] tensor (can be on GPU)
        g_xo_OT:   [N_ot, D] tensor
        h_xs:      [N_hs, D] tensor
        epsilon, tau_a, tau_b: POT hyper-params
        max_workers: # of processes in the pool (None -> os.cpu_count())

        Returns:
        test_transport_matrix: [B, N_hs] tensor on same device as g_xo_test
        """

        # 1) Remember the device, then move once to CPU/NumPy
        device      = g_xo_test.device
        g_xo_test_np = g_xo_test.detach().cpu().numpy()  # [B, D]
        g_xo_OT_np   = g_xo_OT.detach().cpu().numpy()    # [N_ot, D]
        h_xs_np      = h_xs.detach().cpu().numpy()       # [N_hs, D]

        # 2) Build the list of args for each test sample
        args = [
            (g_xo_test_np[i], g_xo_OT_np, h_xs_np, self.epsilon, self.tau_a, self.tau_b )
            for i in range(g_xo_test_np.shape[0])
        ]


        # 3) Parallel map over all CPU cores
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            rows = list(executor.map(OT_align._worker, args))

        # 4) Stack results and send back to the original device
        rows_np = np.stack([np.asarray(r, dtype=np.float32) for r in rows], axis=0)
        test_transport_matrix = torch.tensor(
            rows_np, 
            dtype=torch.float32,
            device=device
            )

        return test_transport_matrix

    
    def eval(self, dataloaders, device):
        self.NPE.eval()
        self.NSE_ft.eval()
        output_dim = len(self.NPE.params_infer)
        xs_loader, test_xo_dataloader, train_xo_loader, val_xo_loader = dataloaders
        h_xs = [] # simulation domain embeddings
        xs_theta_pred = []
        xs_labels = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(xs_loader):
                preds, z = self.NPE(inputs.to(device), 'eval')
                h_xs.append(z)
                xs_theta_pred.append(preds)
                labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
                scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
                scaled_labels = torch.stack(scaled_labels, dim=1)
                xs_labels.append(scaled_labels)
            h_xs = torch.cat(h_xs, dim=0)
            xs_theta_pred = torch.cat(xs_theta_pred, dim=0)
            xs_labels = torch.cat(xs_labels, dim=0)
            g_xo = [] # real domain embeddings
            xo_labels = []
            h_xo = []
            xo_labels_true = []

            for i, (inputs, labels) in enumerate(test_xo_dataloader):

                g_xo.append(self.NSE_ft.get_embedding(inputs.to(device)))

                h_xo.append(self.NPE.get_embedding(inputs.to(device))) 
                labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
                scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
                scaled_labels = torch.stack(scaled_labels, dim=1)
                xo_labels.append(scaled_labels)
                xo_labels_true.append(labels)
            g_xo = torch.cat(g_xo, dim=0)
            xo_labels = torch.cat(xo_labels, dim=0)
            n_o = g_xo.size(0)
            h_xo = torch.cat(h_xo, dim=0)
            xo_labels_true = torch.cat(xo_labels_true, dim=0)

            g_xo_train = []
            h_xo_train = []
            for i, (inputs, labels) in enumerate(train_xo_loader):
                g_xo_train.append(self.NSE_ft.get_embedding(inputs.to(device)))
                h_xo_train.append(self.NPE.get_embedding(inputs.to(device)))
            
            for i, (inputs, labels) in enumerate(val_xo_loader):
                g_xo_train.append(self.NSE_ft.get_embedding(inputs.to(device)))
                h_xo_train.append(self.NPE.get_embedding(inputs.to(device)))
            
            g_xo_train = torch.cat(g_xo_train, dim=0)
            h_xo_train = torch.cat(h_xo_train, dim=0)

            lpp_prior, log_probs_prior = compute_prior_lpp(xo_labels_true, self.params_infer)
            metrics = {'lpp_prior':lpp_prior, 'acauc_prior':0}

            
            transport_matrix = self.compute_single_sample_OT(g_xo, g_xo_train, h_xs)
            transport_matrix_noFT = self.compute_single_sample_OT(h_xo, h_xo_train, h_xs)
            transport_matrix_transductive = self.compute_transport_matrix(g_xo, h_xs)
            transport_matrix_noFT_transductive = self.compute_transport_matrix(h_xo, h_xs)
            preds = torch.matmul(transport_matrix*n_o, xs_theta_pred)
            lpp, log_probs = compute_lpp(self.NPE.predictor, xo_labels, h_xs.detach(), transport_plan=transport_matrix)
            posterior_samples = get_ot_weighted_posterior_samples(self.NPE.predictor, output_dim, h_xs, transport_matrix*n_o, self.posterior_test_sample)
            acauc = compute_acauc_for_samples(posterior_samples, xo_labels, num_alpha=100)
            metrics.update({'lpp':lpp, 'acauc':acauc})
            lpp_npe, log_probs_npe = compute_lpp(self.NPE.predictor, xo_labels, h_xo.detach())
            posterior_samples_npe = self.NPE.predictor.sample(self.posterior_test_sample, context=h_xo.detach())
            acauc_npe = compute_acauc_for_samples(posterior_samples_npe, xo_labels, num_alpha=100)
            metrics_npe = {'lpp_npe':lpp_npe, 'acauc_npe':acauc_npe}
            metrics.update(metrics_npe)
            posterior_samples_finetune = self.NPE.predictor.sample(self.posterior_test_sample, context=g_xo.detach())
            lpp_finetune, log_probs_finetune = compute_lpp(self.NPE.predictor, xo_labels, g_xo.detach())
            acauc_finetune = compute_acauc_for_samples(posterior_samples_finetune, xo_labels, num_alpha=100)
            metrics_finetune = {'lpp_finetune_only':lpp_finetune, 'acauc_finetune_only':acauc_finetune}
            metrics.update(metrics_finetune)
            posterior_samples_sbi = self.NPE.predictor.sample(self.posterior_test_sample, context=h_xs.detach())
            lpp_sbi, log_probs_sbi = compute_lpp(self.NPE.predictor, xs_labels, h_xs.detach())
            acauc_sbi = compute_acauc_for_samples(posterior_samples_sbi, xs_labels, num_alpha=100)
            metrics_sbi = {'lpp_sbi':lpp_sbi, 'acauc_sbi':acauc_sbi}
            metrics.update(metrics_sbi)
            posterior_samples_noFT = get_ot_weighted_posterior_samples(self.NPE.predictor, output_dim, h_xs, transport_matrix_noFT*n_o, self.posterior_test_sample)
            lpp_noFT, log_probs_noFT = compute_lpp(self.NPE.predictor, xo_labels, h_xs.detach(), transport_plan=transport_matrix_noFT)
            acauc_noFT = compute_acauc_for_samples(posterior_samples_noFT, xo_labels, num_alpha=100)
            metrics_noFT = {'lpp_OT_only':lpp_noFT, 'acauc_OT_only':acauc_noFT}
            metrics.update(metrics_noFT)
            posterior_samples_noFT_transductive = get_ot_weighted_posterior_samples(self.NPE.predictor, output_dim, h_xs, transport_matrix_noFT_transductive*n_o, self.posterior_test_sample)
            lpp_noFT_transductive, log_probs_noFT_transductive = compute_lpp(self.NPE.predictor, xo_labels, h_xs.detach(), transport_plan=transport_matrix_noFT_transductive)
            acauc_noFT_transductive = compute_acauc_for_samples(posterior_samples_noFT_transductive, xo_labels, num_alpha=100)
            metric_noFT_transductive = {'lpp_OT_only_transductive':lpp_noFT_transductive, 'acauc_OT_only_transductive':acauc_noFT_transductive}
            metrics.update(metric_noFT_transductive)
            posterior_samples_transductive = get_ot_weighted_posterior_samples(self.NPE.predictor, output_dim, h_xs, transport_matrix_transductive*n_o, self.posterior_test_sample)
            lpp_transductive, log_probs_transductive = compute_lpp(self.NPE.predictor, xo_labels, h_xs.detach(), transport_plan=transport_matrix_transductive)
            metrics_transductive = {'lpp_transductive':lpp_transductive, 'acauc_transductive':acauc_noFT_transductive}
            metrics.update(metrics_transductive)


            param_names = list(self.params_infer.keys())
            corner_sbi = safe_draw_corner(posterior_samples_sbi.detach().cpu().numpy(), xs_labels.detach().cpu().numpy(), log_probs_sbi, list(self.params_infer.keys()), 'SBI')
            corner_npe = safe_draw_corner(posterior_samples_npe.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs_npe, list(self.params_infer.keys()), 'NPE')
            corner_ft = safe_draw_corner(posterior_samples_finetune.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs_finetune, list(self.params_infer.keys()), 'Finetune Only') 
            corner_ot = safe_draw_corner(posterior_samples_noFT.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs_noFT, list(self.params_infer.keys()), 'OT Only')
            corner_rope = safe_draw_corner(posterior_samples.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs, list(self.params_infer.keys()), 'RoPE')
            corner_ot_trans = safe_draw_corner(posterior_samples_noFT_transductive.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs_noFT_transductive, list(self.params_infer.keys()), 'OT Only Transductive')
            corner_rope_trans = safe_draw_corner(posterior_samples_transductive.detach().cpu().numpy(), xo_labels.detach().cpu().numpy(), log_probs_transductive, list(self.params_infer.keys()), 'RoPE Transductive') 
            corner_plots = {'corner_sbi':corner_sbi, 'corner_npe':corner_npe, 'corner_ft':corner_ft, 'corner_ot':corner_ot, 'corner_rope':corner_rope, 'corner_ot_trans':corner_ot_trans, 'corner_rope_trans':corner_rope_trans}
        return corner_plots, metrics


class NF_OT_align(BaseStage):
    def __init__(self, backbone, params_infer, args, posterior_train_sample=100, posterior_test_sample=1000, fold=None):
        super(NF_OT_align, self).__init__()
        self.params_infer = params_infer
        num_param = len(params_infer)
        self.label_idx = [val for key, val in params_infer.items()]
        self.NPE = None
        self.NSE_ft = None 
        self.output_size = backbone['params']['output_size']
        self.predictor = ConditionalMAF(input_dim=num_param, context_dim=self.output_size)
        self.fold = fold
        self.posterior_train_sample = posterior_train_sample
        self.posterior_test_sample = posterior_test_sample
        self.epsilon = args['gamma']
        self.tau_a = args['tau_a']
        self.tau_b = args['tau_b']
    
    def prepare_data_for_OT_(self, train_xo_dataloader, val_xo_dataloader, xs_dataloader, device):
        h_xs = []
        xs_theta_pred = []
        xs_theta_log_probs = []
        for i, (inputs, labels) in enumerate(xs_dataloader):
            _ , z = self.NPE(inputs.to(device), 'eval')
            h_xs.append(z)
            preds = self.NPE.predictor.sample(100, context=z)
            xs_theta_pred.append(preds)
            labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
            scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
            scaled_labels = torch.stack(scaled_labels, dim=1).to(device)
            xs_theta_log_probs.append(self.NPE.predictor(scaled_labels, context=z))
        h_xs = torch.cat(h_xs, dim=0)
        xs_theta_pred = torch.cat(xs_theta_pred, dim=0)
        xs_theta_log_probs = torch.cat(xs_theta_log_probs, dim=0)
        g_xo = []
        g_xo_train = []
        xo_train_labels = []

        for i, (inputs, labels) in enumerate(train_xo_dataloader):
            g_xo_train.append(self.NSE_ft.get_embedding(inputs.to(device)))
            labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
            xo_train_labels.append(labels)
        g_xo_train = torch.cat(g_xo_train, dim=0)
        xo_train_labels = torch.cat(xo_train_labels, dim=0)
        id_xo_train = torch.arange(g_xo_train.size(0))

        g_xo_val = []
        xo_val_labels = []


        for i, (inputs, labels) in enumerate(val_xo_dataloader):
            g_xo_val.append(self.NSE_ft.get_embedding(inputs.to(device)))
            labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
            xo_val_labels.append(labels)
        g_xo_val = torch.cat(g_xo_val, dim=0)
        xo_val_labels = torch.cat(xo_val_labels, dim=0)
        id_xo_val = torch.arange(g_xo_val.size(0))
        self.n_o = g_xo_train.size(0) + g_xo_val.size(0)

        return h_xs.detach(), xs_theta_pred.detach(), xs_theta_log_probs.detach(), (g_xo_train.detach().cpu(), id_xo_train.detach().cpu(), xo_train_labels.detach().cpu()),  (g_xo_val.detach().cpu(), id_xo_val.detach().cpu(), xo_val_labels.detach().cpu())
    
    def compute_transport_matrix_(self, train_xo_dataloader, val_xo_dataloader, xs_dataloader, device):
        h_xs, xs_theta_pred, xs_theta_log_probs, g_xo_train, g_xo_val = self.prepare_data_for_OT_(train_xo_dataloader, val_xo_dataloader, xs_dataloader, device)
        g_xo = torch.cat([g_xo_train[0], g_xo_val[0]], dim=0)
        transport_matrix, _ = compute_semi_balanced_ot(g_xo.detach().cpu().numpy(), h_xs.detach().cpu().numpy(), epsilon=self.epsilon,  tau_a=self.tau_a, tau_b=self.tau_b)
        transport_matrix = torch.tensor(transport_matrix.tolist(), dtype=torch.float32)
        self.transport_matrix = transport_matrix.to(device)
        self.alpha_ij = self.transport_matrix*self.n_o
        self.xs_theta_pred = xs_theta_pred
        self.xs_theta_log_probs = xs_theta_log_probs
        return g_xo_train, g_xo_val

    def forward(self, inputs, phase = 'train'):
        '''
        inputs: tuple of (g_xo_i, i)
            g_xo_i: embedding of the real domain data (n,d)
            i: index of the transport matrix row (n,)
        '''
        if phase == 'train':
            self.predictor.train()
        else:
            self.predictor.eval()
        
        if phase == 'train' or phase == 'eval':
            num_samples = self.posterior_train_sample
            g_xo, _ = torch.split(inputs, [inputs.size(1)-1, 1], dim=-1)
            xs_theta_pred_expand = self.xs_theta_pred.unsqueeze(0).expand(g_xo.size(0), -1, -1, -1)
            g_xo_expand = g_xo.unsqueeze(1).unsqueeze(1).expand(-1, self.xs_theta_pred.size(0),self.xs_theta_pred.size(1), -1)
            log_probs = self.predictor(xs_theta_pred_expand.reshape(-1, xs_theta_pred_expand.size(-1)), context=g_xo_expand.reshape(-1, g_xo_expand.size(-1)))
            log_probs = log_probs.view(g_xo.size(0), self.xs_theta_pred.size(0), self.xs_theta_pred.size(1))
            #log_probs = None
        else:
            num_samples = self.posterior_test_sample
            self.NSE_ft.eval()
            g_xo = self.NSE_ft.get_embedding(inputs)
            log_probs = None
        posterior_samples = self.predictor.sample(num_samples, context=g_xo)
        return log_probs, (g_xo, posterior_samples)

    def data_preprocess(self, inputs, labels):
        return inputs, labels

    def metrics_calculate(self, data, phase='train'):
        if phase == 'train' or phase == 'eval':
            log_probs, _, inputs,  _ = data
            _, i = torch.split(inputs, [inputs.size(1)-1, 1], dim=-1)
            i = i.long().squeeze().cpu().detach()
            alpha_ij = self.alpha_ij[i, :].unsqueeze(-1).expand(-1, -1, self.xs_theta_pred.size(1))
            log_probs_weighted = torch.sum(alpha_ij * log_probs, dim=1)
            loss = -torch.mean(log_probs_weighted)
            metrics = {'loss':loss}
        else:
            _, preds, _, labels = data
            labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
            scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
            scaled_labels = torch.stack(scaled_labels, dim=1)
            g_xo, posterior_samples = preds
            lpp, log_probs = compute_lpp(self.predictor, scaled_labels, g_xo.detach())
            acauc = compute_acauc_for_samples(posterior_samples, scaled_labels, num_alpha=100)
            metrics = {'lpp':lpp, 'acauc':acauc}
            corner_amortised_OT = safe_draw_corner(posterior_samples.detach().cpu().numpy(), scaled_labels.detach().cpu().numpy(), log_probs, list(self.params_infer.keys()),'amortised OT solution') #, f"corner_amortised_OT_{'_'.join(self.params_infer.keys())}")
            corner_plots = {'corner_amortised_OT':corner_amortised_OT}
            metrics = (corner_plots, metrics)
            
        return metrics

    def train_step(self, inputs, labels):
        inputs, labels = self.data_preprocess(inputs, labels)
        log_probs, _ = self.forward(inputs, 'train')
        metrics = self.metrics_calculate((log_probs, None, inputs, None), phase='train')
        return None, metrics

    def eval_step(self, inputs, labels, eval_phase='eval'):
        inputs, labels = self.data_preprocess(inputs, labels)
        with torch.no_grad():
            log_probs, preds = self.forward(inputs, eval_phase)
            metrics = self.metrics_calculate((log_probs, preds, inputs, labels), phase=eval_phase)
        return preds, metrics
    
    def optimise(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Wass_OT_align_finetune(NSE_finetune):
    def __init__(self, backbone, params_infer, gamma, posterior_train_sample, posterior_test_sample, fold=None):
        super(Wass_OT_align_finetune, self).__init__(backbone, params_infer, posterior_train_sample, posterior_test_sample, fold)
        self.gamma = gamma

    def prepare_data_for_OT_(self, xs_dataloader, device):
        h_xs = []
        xs_theta_pred = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(xs_dataloader):
                _, z = self.NPE(inputs.to(device), 'eval')
                h_xs.append(z)
                preds = self.NPE.predictor.sample(100, context=z)
                xs_theta_pred.append(preds)
            h_xs = torch.cat(h_xs, dim=0)
            xs_theta_pred = torch.cat(xs_theta_pred, dim=0)
        self.h_xs = h_xs.detach()
        self.xs_theta_pred = xs_theta_pred.detach()
    
    def prepare_supervised_targets(self, train_calib_dataloader, val_calib_dataloader, device):
        inputs_train = []
        target_preds_gt_train = []
        inputs_val = []
        target_preds_gt_val = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(train_calib_dataloader):
                inputs_train.append(inputs)
                target_preds_gt = self.compute_zs_gt(labels)
                target_preds_gt_train.append(target_preds_gt)
            for i, (inputs, labels) in enumerate(val_calib_dataloader):
                inputs_val.append(inputs)
                target_preds_gt = self.compute_zs_gt(labels)
                target_preds_gt_val.append(target_preds_gt)
            inputs_train = torch.cat(inputs_train, dim=0)
            inputs_val = torch.cat(inputs_val, dim=0)
            target_preds_gt_train = torch.cat(target_preds_gt_train, dim=0)
            target_preds_gt_val = torch.cat(target_preds_gt_val, dim=0)

        self.train_calib_inputs = inputs_train
        self.train_calib_labels = target_preds_gt_train
        self.val_calib_inputs = inputs_val
        self.val_calib_labels = target_preds_gt_val


    def train_step(self, inputs, labels):
        inputs, labels = self.data_preprocess((inputs, labels))
        preds = self.forward(inputs, 'train')
        metrics = self.metrics_calculate((preds, labels), phase='train')
        return preds, metrics
    def eval_step(self, inputs, labels):
        inputs, labels = self.data_preprocess((inputs, labels))
        with torch.no_grad():
            preds = self.forward(inputs, 'eval')
            metrics = self.metrics_calculate((preds, labels), phase='eval')
        return preds, metrics

    
    def metrics_calculate(self, data, phase='train'):
        preds, _ = data
        preds, z = preds
        if phase == 'train': 
            input_calib = self.train_calib_inputs
            labels_calib = self.train_calib_labels
        else:
            input_calib = self.val_calib_inputs
            labels_calib = self.val_calib_labels
        z_calib = self.encoder(input_calib.to(z.device))
        metrics = {}
        sup_loss = self.mse_loss(z_calib, labels_calib)  
        unified_h_xs = torch.cat([self.h_xs, labels_calib], dim=0)
        loss_ot, loss_entropy, alpha = unbalanced_ot_wasserstine_loss(torch.cat([z,z_calib], dim=0), unified_h_xs, self.gamma)
        metrics['supervised_mse'] = sup_loss
        metrics['loss_wass_ot'] = loss_ot
        metrics['loss_entropy'] = loss_entropy
        total_loss = sup_loss+loss_ot
        metrics['loss'] = total_loss
        return metrics
    
    def calculate_final_alpha(self, inputs):
        with torch.no_grad():
            z = self.encoder(inputs)
            _, _, alpha = unbalanced_ot_wasserstine_loss(z, self.h_xs, self.gamma)
        return alpha.detach()

class NF_wassOT_align(NF_OT_align):
    def __init__(self, backbone, params_infer, args, posterior_train_sample=100, posterior_test_sample=1000, fold=None):
        super(NF_wassOT_align, self).__init__(backbone, params_infer, args, posterior_train_sample, posterior_test_sample, fold)
        self.wass_OT = None

    def metrics_calculate(self, data, phase='train'):
        if phase == 'train' or phase == 'eval':
            log_probs, _, inputs,  _ = data
            alpha_ij = self.wass_OT.calculate_final_alpha(inputs)
            alpha_ij = alpha_ij.unsqueeze(-1).expand(-1, -1, self.wass_OT.xs_theta_pred.size(1))
            log_probs_weighted = torch.sum(alpha_ij * log_probs, dim=1)
            loss = -torch.mean(log_probs_weighted)
            metrics = {'loss':loss}
        else:
            _, preds, inputs, labels = data
            labels = torch.cat([labels[param['index']].unsqueeze(1) for param in self.label_idx], dim=1)
            scaled_labels = [self.scale_params(labels[:, ind], name) for ind, name in enumerate(self.params_infer)]
            scaled_labels = torch.stack(scaled_labels, dim=1)
            g_xo, posterior_samples = preds
            lpp, log_probs = compute_lpp(self.predictor, scaled_labels, g_xo.detach())
            acauc = compute_acauc_for_samples(posterior_samples, scaled_labels, num_alpha=100)
            corner_amortised_end2end = safe_draw_corner(posterior_samples.detach().cpu().numpy(), scaled_labels.detach().cpu().numpy(), log_probs, list(self.params_infer.keys()), 'amortised RoPE end-to-end') #, f"corner_end2end_{'_'.join(self.params_infer.keys())}")
            metrics = {'lpp':lpp, 'acauc':acauc}
            alpha_ij = self.wass_OT.calculate_final_alpha(inputs)
            output_dim = len(self.params_infer)
            posterior_samples_wass = get_ot_weighted_posterior_samples(self.wass_OT.NPE.predictor, output_dim, self.wass_OT.h_xs, alpha_ij, self.posterior_test_sample)
            n_o = self.wass_OT.h_xs.size(0)
            lpp_wass, log_probs_wass = compute_lpp(self.wass_OT.NPE.predictor, scaled_labels, self.wass_OT.h_xs.detach(), transport_plan=alpha_ij/n_o)
            acauc_wass = compute_acauc_for_samples(posterior_samples_wass, scaled_labels, num_alpha=100)
            metrics_wass = {'lpp_wass':lpp_wass, 'acauc_wass':acauc_wass}
            metrics.update(metrics_wass)
            corner_wass_end2end = safe_draw_corner(posterior_samples_wass.detach().cpu().numpy(), scaled_labels.detach().cpu().numpy(), log_probs_wass, list(self.params_infer.keys()), 'amortised RoPE end-to-end wass2MSE') #, f"corner_end2end_wass_{'_'.join(self.params_infer.keys())}")
            corner_plots = {'corner_amortised_end2end':corner_amortised_end2end, 'corner_wass_end2end':corner_wass_end2end}
            metrics = (corner_plots, metrics)
        return metrics

    def forward(self, inputs, phase = 'train'):
        '''
        inputs: tuple of (g_xo_i, i)
            g_xo_i: embedding of the real domain data (n,d)
            i: index of the transport matrix row (n,)
        '''
        if phase == 'train':
            self.predictor.train()
        else:
            self.predictor.eval()
        
        with torch.no_grad():
            g_xo = self.wass_OT.encoder(inputs).detach()
        if phase == 'train' or phase == 'eval':
            num_samples = self.posterior_train_sample
            xs_theta_pred_expand = self.wass_OT.xs_theta_pred.unsqueeze(0).expand(g_xo.size(0), -1, -1, -1)
            g_xo_expand = g_xo.unsqueeze(1).unsqueeze(1).expand(-1, self.wass_OT.xs_theta_pred.size(0),self.wass_OT.xs_theta_pred.size(1), -1)
            log_probs = self.predictor(xs_theta_pred_expand.reshape(-1, xs_theta_pred_expand.size(-1)), context=g_xo_expand.reshape(-1, g_xo_expand.size(-1)))
            log_probs = log_probs.view(g_xo.size(0), self.wass_OT.xs_theta_pred.size(0), self.wass_OT.xs_theta_pred.size(1))
        else:
            num_samples = self.posterior_test_sample
            log_probs = None
        posterior_samples = self.predictor.sample(num_samples, context=g_xo)
        return log_probs, (g_xo, posterior_samples)
    
    
    



class RoPE(BasePipeline):
    def __init__(self, args, wandb_logger, device, fold=None):
        self.args = args
        self.wandb_logger = wandb_logger
        self.epochs = args['training']['epochs']
        self.backbone = args['backbone']
        self.params_infer = args['params_infer']
        self.posterior_train_sample = args['training']['posterior_train_sample']
        self.posterior_test_sample = args['training']['posterior_test_sample']
        self.NPE = RoPE_NPE(self.backbone, self.params_infer, self.posterior_train_sample, self.posterior_test_sample).to(device)
        self.NSE_ft = NSE_finetune(self.backbone, self.params_infer,  self.posterior_train_sample, self.posterior_test_sample, fold).to(device)
        self.OT_align = OT_align(args['params_infer'], args['training'], self.posterior_train_sample, self.posterior_test_sample)
        self.device = device
        self.early_stopping = args['training']['early_stopping']
        self.patience = args['training']['patience']
    
    def set_optimisers(self):
        lr_npe = self.args['training']['lr_npe']
        lr_nse_ft = self.args['training']['lr_nse_ft']
        self.NPE.optimizer = torch.optim.Adam(self.NPE.parameters(), lr=lr_npe)
        self.NSE_ft.optimizer = torch.optim.Adam(self.NSE_ft.encoder.parameters(), lr=lr_nse_ft)

    
    def train_stage(self, stage, train_dataloader, val_dataloader, early_stopping=True, patience=5, pretrained_encoder=None):
        stage, name = stage
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            metrics_val_accum = {}
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_dataloader):
                    inputs, labels = self.mount_to_device(inputs), self.mount_to_device(labels)
                    preds, metrics = stage.eval_step(inputs, labels)
                    metrics_val_accum = {key:metrics_val_accum.get(key, 0) + metrics[key] for key in metrics}
                metrics = {key:val/len(val_dataloader) for key, val in metrics_val_accum.items()}
            if metrics['loss'] < best_val_loss:
                best_val_loss = metrics['loss']
                epochs_no_improve = 0
                stage.save_checkpoint(self.wandb_logger.id)
            else:
                epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                break
            self.wandb_log_metrics(metrics, f"{name}_val") 
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = self.mount_to_device(inputs), self.mount_to_device(labels)
                preds, metrics = stage.train_step(inputs, labels)
                loss = metrics['loss']
                self.wandb_log_metrics(metrics, f"{name}_train")
                stage.optimise(loss)
        stage.load_checkpoint(self.wandb_logger.id)
        return best_val_loss


    def train_pipeline(self, dataloaders):
        self.set_optimisers()
        train_xs_npe_dataloader, val_xs_npe_dataloader, train_xo_xs_xfa_dataloader, val_xo_xs_xfa_dataloader, train_xo_calib_dataloader, val_xo_calib_dataloader = dataloaders
        if not self.NPE.load_checkpoint(self.args['training']['npe_ckpt']):
            self.train_stage((self.NPE, 'NPE'), train_xs_npe_dataloader, val_xs_npe_dataloader, early_stopping=self.early_stopping[0], patience=self.patience[0])
        self.NSE_ft.simulate_ = train_xs_npe_dataloader.dataset.simulate_ 
        self.NSE_ft.pretrained_encoder = self.NPE.encoder
        if not self.NSE_ft.load_checkpoint(self.args['training']['nse_ckpt']):
            self.NSE_ft.encoder.load_state_dict(self.NPE.encoder.state_dict())
            self.train_stage((self.NSE_ft, 'NSE_ft'), train_xo_calib_dataloader, val_xo_calib_dataloader, early_stopping=self.early_stopping[1], patience=self.patience[1])
        return 0.0
        
    
    def test_pipeline(self, dataloaders, ckpt_path=None):
        if ckpt_path:
            self.NPE.load_checkpoint(ckpt_path)
            self.NSE_ft.encoder.load_checkpoint(ckpt_path)
        self.OT_align.NPE = self.NPE
        self.OT_align.NSE_ft = self.NSE_ft
        corner_plots, metrics = self.OT_align.eval(dataloaders, self.device)
        self.wandb_log_metrics(metrics, f"OT_align_test")
        self.wandb_logger.log({name: wandb.Image(fig) for name, fig in corner_plots.items()})
        return metrics

class NFRoPE(RoPE):
    def __init__(self, args, wandb_logger, device, fold=None):
        super().__init__(args, wandb_logger, device, fold)
        self.NF_align = NF_OT_align(self.backbone, self.params_infer, args['training'], self.posterior_train_sample, self.posterior_test_sample, fold ).to(device)
        self.load_encoders_ckpt_()
        
    
    def load_encoders_ckpt_(self):
        if self.args['training']['npe_ckpt']:
            self.NPE.load_checkpoint(self.args['training']['npe_ckpt'])
        else:
            self.NPE.load_checkpoint(self.wandb_logger.id)

        self.NSE_ft.pretrained_encoder = self.NPE.encoder
        if self.args['training']['nse_ckpt']:
            self.NSE_ft.load_checkpoint(self.args['training']['nse_ckpt'])
        else:
            self.NSE_ft.load_checkpoint(self.wandb_logger.id)


    def train_pipeline(self, dataloaders):
        self.set_optimisers()
        train_xs_npe_dataloader, _, train_xo_dataloader, val_xo_dataloader, _, _,  xs_test_loader = dataloaders
        self.NF_align.NPE = self.NPE
        self.NF_align.NSE_ft = self.NSE_ft
        if not self.NF_align.load_checkpoint(self.args['training']['nf_align_ckpt']):
            with torch.no_grad():
                g_xo_train, g_xo_val = self.NF_align.compute_transport_matrix_(train_xo_dataloader, val_xo_dataloader, xs_test_loader, self.device)
            g_xo_train_dataset = TensorDataset(torch.cat((g_xo_train[0], g_xo_train[1].unsqueeze(1)), dim=-1), g_xo_train[2])
            g_xo_val_dataset = TensorDataset(torch.cat((g_xo_val[0], g_xo_val[1].unsqueeze(1)), dim=-1), g_xo_val[2])
            train_xo_dataloader = DataLoader(g_xo_train_dataset, batch_size=train_xo_dataloader.batch_size, shuffle=True)
            val_xo_dataloader = DataLoader(g_xo_val_dataset, batch_size=val_xo_dataloader.batch_size, shuffle=False)
            self.train_stage((self.NF_align, 'NF_align'), train_xo_dataloader, val_xo_dataloader, early_stopping=self.early_stopping[2], patience=self.patience[2])

    def test_pipeline(self, dataloaders, ckpt_path=None):
        if ckpt_path:
            self.NPE.load_checkpoint(ckpt_path)
            self.NSE_ft.encoder.load_checkpoint(ckpt_path)
            self.NF_align.load_checkpoint(ckpt_path)
        self.NF_align.NPE = self.NPE
        self.NF_align.NSE_ft = self.NSE_ft
        test_metrics = {}
        inputs_list = []
        labels_list = []
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs_list.append(inputs)
            labels_list.append(labels)
        inputs = torch.cat(inputs_list, dim=0)
        unzipped = list(zip(*labels_list))
        labels = [ torch.stack([item for sublist in group for item in sublist], axis=0) for group in unzipped ]
        labels = tuple(labels)
        inputs, labels = self.mount_to_device(inputs), self.mount_to_device(labels)
        _, test_metrics = self.NF_align.eval_step(inputs, labels, eval_phase='test')
        corner_plots, test_metrics = test_metrics
        self.wandb_log_metrics(test_metrics, f"NF_align_test")
        self.wandb_logger.log({name: wandb.Image(fig) for name, fig in corner_plots.items()})
        return test_metrics
    
    def set_optimisers(self):
        lr_nf_align = self.args['training']['lr_nf_align']
        self.NF_align.optimizer = torch.optim.Adam(self.NF_align.parameters(), lr=lr_nf_align)

class WassRoPE(RoPE):
    def __init__(self, args, wandb_logger, device, fold=None):
        super().__init__(args, wandb_logger, device, fold)
        self.Wass_OT_align_finetune = Wass_OT_align_finetune(self.backbone, self.params_infer, args['training']['gamma'], self.posterior_train_sample, self.posterior_test_sample, fold).to(device)
        self.NF_wassOT_align = NF_wassOT_align(self.backbone, self.params_infer, args['training'], self.posterior_train_sample, self.posterior_test_sample, fold).to(device)
        self.load_encoders_ckpt_()
        self.Wass_OT_align_finetune.pretrained_encoder = self.NPE.encoder

    
    def load_encoders_ckpt_(self):
        if self.args['training']['npe_ckpt']:
            self.NPE.load_checkpoint(self.args['training']['npe_ckpt'])
        else: 
            self.NPE.load_checkpoint(self.wandb_logger.id)
    
    def set_optimisers(self):
        lr_nse_ft_e2e = self.args['training']['lr_nse_ft_end_to_end']
        self.Wass_OT_align_finetune.optimizer = torch.optim.Adam(self.Wass_OT_align_finetune.encoder.parameters(), lr=lr_nse_ft_e2e)
        lr_nf_wassOT_align = self.args['training']['lr_nf_align']
        self.NF_wassOT_align.optimizer = torch.optim.Adam(self.NF_wassOT_align.predictor.parameters(), lr=lr_nf_wassOT_align)

    def train_pipeline(self, dataloaders):
        best_val_loss = float('inf')
        self.set_optimisers()
        train_xo_dataloader, val_xo_dataloader, xs_test_loader, xo_calib_train_loader, xo_calib_val_loader = dataloaders
        train_xo_dataloader, train_xo_dataloader_small_batch = train_xo_dataloader
        val_xo_dataloader, val_xo_dataloader_small_batch = val_xo_dataloader
        self.Wass_OT_align_finetune.simulate_ = xs_test_loader.dataset.simulate_ 
        self.Wass_OT_align_finetune.NPE = self.NPE
        self.Wass_OT_align_finetune.prepare_supervised_targets(xo_calib_train_loader, xo_calib_val_loader, self.device)
        with torch.no_grad():
            self.Wass_OT_align_finetune.prepare_data_for_OT_(xs_test_loader, self.device)
        if not self.Wass_OT_align_finetune.load_checkpoint(self.args['training']['wassOT_ft_ckpt']):
            self.Wass_OT_align_finetune.encoder.load_state_dict(self.NPE.encoder.state_dict())
            best_val_loss = self.train_stage((self.Wass_OT_align_finetune, 'Wass_OT_align_finetune'), train_xo_dataloader, val_xo_dataloader, early_stopping=self.early_stopping[1], patience=self.patience[1])
        self.NF_wassOT_align.NPE = self.NPE
        self.NF_wassOT_align.wass_OT = self.Wass_OT_align_finetune
        if not self.NF_wassOT_align.load_checkpoint(self.args['training']['nf_wassOT_align_ckpt']):
            self.NF_wassOT_align.wass_OT = self.Wass_OT_align_finetune
            self.NF_wassOT_align.NPE = self.NPE
            self.train_stage((self.NF_wassOT_align, 'NF_wassOT_align'), train_xo_dataloader_small_batch, val_xo_dataloader_small_batch, early_stopping=self.early_stopping[2], patience=self.patience[2])
        return best_val_loss
    
    def test_pipeline(self, dataloaders, ckpt_path=None):
        if ckpt_path:
            self.Wass_OT_align_finetune.load_checkpoint(ckpt_path)
            self.NF_wassOT_align.load_checkpoint(ckpt_path)
        test_metrics = {}
        inputs_list = []
        labels_list = []
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs_list.append(inputs)
            labels_list.append(labels)
        inputs = torch.cat(inputs_list, dim=0)
        unzipped = list(zip(*labels_list))
        labels = [ torch.stack([item for sublist in group for item in sublist], axis=0) for group in unzipped ]
        labels = tuple(labels)
        inputs, labels = self.mount_to_device(inputs), self.mount_to_device(labels)
        _, test_metrics = self.NF_wassOT_align.eval_step(inputs, labels, eval_phase='test')
        corner_plots, test_metrics = test_metrics
        self.wandb_log_metrics(test_metrics, f"WassOT_NF_align_test")
        self.wandb_logger.log({name: wandb.Image(fig) for name, fig in corner_plots.items()})
        return test_metrics