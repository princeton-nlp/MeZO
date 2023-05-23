from transformers.utils import logging
import torch
import torch.nn.functional as F
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegressionCV
logger = logging.get_logger(__name__)

class BaseKernelSolver:
    def __init__(self, args):
        self.args = args

        # initialized with fit()
        self.num_labels = None
        self.kernel_dtype = None

    def get_regularized_kernel(self, kernel):
        if self.args.kernel_regularization:
            _, S, _ = torch.svd(kernel)
            op_norm = S.max()

            reg = (self.args.kernel_regularization * op_norm).to(kernel.dtype)

            identity = torch.eye(kernel.shape[1], dtype=kernel.dtype).unsqueeze(0)
            return kernel + identity * reg
        else:
            return kernel

    def metrics(self):
        return {}

    def get_target_coords(self, targets):
        if self.num_labels == 1:  # Regression
            return targets.flatten().to(self.kernel_dtype).unsqueeze(1)
        else:
            return torch.nn.functional.one_hot(targets.flatten(), self.num_labels).to(self.kernel_dtype)

    def loss(self, preds, targets):
        targets_coords = self.get_target_coords(targets)
        return 1/targets_coords.shape[0] * ((preds - targets_coords)**2).sum().item()

    def fit(self, train_kernel, train_targets, train_logits=None):
        raise NotImplementedError("BaseKernelSolver is just the abstract base class")

    def predict(self, eval_kernel, eval_targets, eval_logits=None):
        raise NotImplementedError("BaseKernelSolver is just the abstract base class")


class LstsqKernelSolver(BaseKernelSolver):
    def __init__(self, args):
        super().__init__(args)

        # initialized with fit()
        self.kernel_solution = None
        self.residual = None
        self.rank = None

    def metrics(self):
        metrics_dict = super().metrics()
        if self.rank is not None:
            if self.rank.numel() > 1:
                for i,r in enumerate(self.rank.tolist()):
                    metrics_dict["rank{}".format(i)] = r
            else:
                metrics_dict["rank0"] = self.rank.item()
        return metrics_dict

    def fit(self, train_kernel, train_targets, train_logits=None):
        self.num_labels = train_kernel.size(0)
        self.kernel_dtype = train_kernel.dtype

        kernel = self.get_regularized_kernel(train_kernel)
        train_targets_coords = self.get_target_coords(train_targets)

        if train_logits is not None and self.args.f0_scaling > 0:
            train_targets_coords -= train_logits / self.args.f0_scaling

        self.kernel_solution, self.residuals, self.rank, _ = torch.linalg.lstsq(kernel, train_targets_coords.t())

    def predict(self, eval_kernel, eval_targets, eval_logits=None, **unused_kwargs):
        assert self.kernel_solution is not None, "Must call fit() before predict()"
        assert eval_kernel.size(0) == self.num_labels, "Number of labels in eval_kernel must match fit()"

        eval_preds = torch.bmm(
            eval_kernel.transpose(1, 2),
            self.kernel_solution.unsqueeze(2)
        ).squeeze(2).transpose(0, 1) # shape [#dataset_outer, #classes]

        if eval_logits is not None and self.args.f0_scaling > 0:
            eval_preds += eval_logits / self.args.f0_scaling

        eval_loss = self.loss(eval_preds, eval_targets)
        return eval_loss, eval_preds


class AsymmetricLstsqKernelSolver(LstsqKernelSolver):
    def __init__(self, args):
        super().__init__(args)

        self.N = None
        self.train_targets = None

    def fit(self, train_kernel, train_targets, train_logits=None):
        self.num_labels = train_kernel.size(0)
        self.kernel_dtype = train_kernel.dtype
        assert self.num_labels == 1, "SVMKernelSolver only works for regression tasks or binary_classification"

        kernel = self.get_regularized_kernel(train_kernel)
        train_targets_coords = self.get_target_coords(train_targets)

        if train_logits is not None and self.args.f0_scaling > 0:
            train_targets_coords -= train_logits / self.args.f0_scaling

        kernel = kernel.squeeze()
        H = torch.zeros(kernel.shape)
        Y = train_targets_coords.squeeze()
        N = H.shape[0]

        for i in range(N):
            for j in range(N):
                H[i,j] = Y[i] * (kernel[i,j]* Y[j])

        # # system with biases
        # A = torch.zeros(2*N + 2, 2*N + 2, dtype=self.kernel_dtype)
        # A[0, 2:2+N] = Y
        # A[1, 2+N:] = Y
        # A[2:2+N, 0] = Y
        # A[2+N:, 1] = Y
        # A[2:2+N, 2:2+N] = torch.eye(N) / self.args.kernel_gamma # scale by 1/gamma later
        # A[2:2+N, 2+N:] = H
        # A[2+N:, 2:2+N] = H.T
        # A[2+N:, 2+N:] = torch.eye(N) / self.args.kernel_gamma # scale by 1/gamma later

        # B = torch.zeros(2*N+2, dtype=self.kernel_dtype)
        # B[2:] = 1

        # system without biases
        A = torch.zeros(2*N, 2*N, dtype=self.kernel_dtype)
        A[:N, :N] = torch.eye(N) / self.args.kernel_gamma # scale by 1/gamma later
        A[:N, N:] = H
        A[N:, :N] = H.T
        A[N:, N:] = torch.eye(N) / self.args.kernel_gamma # scale by 1/gamma later
        B = torch.ones(2*N, dtype=self.kernel_dtype)

        self.N = N
        self.Y = Y
        self.kernel_solution, self.residuals, self.rank, _ = torch.linalg.lstsq(A, B)

    def predict(self, eval_kernel, eval_targets, eval_logits=None, eval_kernel_flipped=None, **unused_kwargs):
        assert self.kernel_solution is not None, "Must call fit() before predict()"
        assert eval_kernel.size(0) == self.num_labels, "Number of labels in eval_kernel must match fit()"

        N = self.N
        # beta_bias = self.kernel_solution[0]
        # alpha_bias = self.kernel_solution[1]
        # alpha = self.kernel_solution[2:2+N].unsqueeze(0)
        # beta = self.kernel_solution[2+N:].unsqueeze(0)
        alpha = self.kernel_solution[:N].unsqueeze(0)
        beta = self.kernel_solution[N:].unsqueeze(0)

        omega = torch.bmm(
            eval_kernel.transpose(1, 2),
            (alpha*self.Y).unsqueeze(2)
        ).squeeze(2).transpose(0, 1) #+ alpha_bias
        nu = torch.bmm(
            eval_kernel_flipped.transpose(1, 2),
            (beta*self.Y).unsqueeze(2)
        ).squeeze(2).transpose(0, 1) #+ beta_bias

        eval_preds = (self.args.kernel_lambda * omega + (1-self.args.kernel_lambda) * nu)

        if eval_logits is not None and self.args.f0_scaling > 0:
            eval_preds += eval_logits / self.args.f0_scaling

        eval_loss = self.loss(eval_preds, eval_targets)
        return eval_loss, eval_preds


class SVRKernelSolver(BaseKernelSolver):
    def __init__(self, args):
        super().__init__(args)

        self.svms = None

    def fit(self, train_kernel, train_targets, train_logits=None):
        self.num_labels = train_kernel.size(0)
        self.kernel_dtype = train_kernel.dtype

        kernel = self.get_regularized_kernel(train_kernel)
        train_targets_coords = self.get_target_coords(train_targets)

        if train_logits is not None and self.args.f0_scaling > 0:
            train_targets_coords -= train_logits / self.args.f0_scaling

        self.svms = []
        for k in range(self.num_labels):
            svm = SVR(kernel='precomputed')
            svm.fit(kernel[k].cpu().numpy(), train_targets_coords[:,k].cpu().t().numpy())
            self.svms.append(svm)

    def predict(self, eval_kernel, eval_targets, eval_logits=None, **unused_kwargs):
        assert self.svms is not None, "Must call fit() before predict()"
        assert eval_kernel.size(0) == self.num_labels, "Number of labels in eval_kernel must match fit()"

        eval_preds = []
        for k in range(self.num_labels):
            predict_k = self.svms[k].predict(eval_kernel[k].cpu().t().numpy())
            eval_preds.append(torch.tensor(predict_k, dtype=self.kernel_dtype, device=eval_kernel.device))
        eval_preds = torch.stack(eval_preds, dim=1)
        print(eval_preds, eval_targets)

        if eval_logits is not None and self.args.f0_scaling > 0:
            eval_preds += eval_logits / self.args.f0_scaling

        eval_loss = self.loss(eval_preds, eval_targets)
        return eval_loss, eval_preds


class SVCKernelSolver(BaseKernelSolver):
    def __init__(self, args):
        super().__init__(args)

        self.svms = None

    def fit(self, train_kernel, train_targets, train_logits=None):
        self.num_labels = train_kernel.size(0)
        self.kernel_dtype = train_kernel.dtype
        assert self.num_labels == 1, "SVMKernelSolver only works for binary_classification"
        assert train_logits is None, "SVMKernelSolver does not support train_logits"

        kernel = self.get_regularized_kernel(train_kernel)
        train_targets = ((train_targets + 1) / 2).int() # convert back from {-1, 1} to {0, 1}

        self.svms = []
        for k in range(self.num_labels):
            svm = SVC(kernel='precomputed')
            svm.fit(kernel[k].cpu().numpy(), train_targets.cpu().numpy())
            self.svms.append(svm)

    def predict(self, eval_kernel, eval_targets, eval_logits=None, **unused_kwargs):
        assert self.svms is not None, "Must call fit() before predict()"
        assert eval_kernel.size(0) == self.num_labels, "Number of labels in eval_kernel must match fit()"
        assert eval_logits is None, "SVMKernelSolver does not support train_logits"

        eval_preds = []
        for k in range(self.num_labels):
            predict_k = self.svms[k].predict(eval_kernel[k].cpu().t().numpy())
            eval_preds.append(torch.tensor(predict_k, dtype=self.kernel_dtype, device=eval_kernel.device))
        eval_preds = torch.stack(eval_preds, dim=1)

        eval_preds = (eval_preds * 2 - 1) # convert back from {0, 1} to {-1, 1}

        eval_loss = self.loss(eval_preds, eval_targets)
        return eval_loss, eval_preds


class LogisticKernelSolver(BaseKernelSolver):
    def __init__(self, args):
        super().__init__(args)

        self.logistic_model = None

    def fit(self, train_kernel, train_targets, train_logits=None):
        self.num_labels = train_kernel.size(0)
        self.kernel_dtype = train_kernel.dtype
        assert self.num_labels == 1, "SVMKernelSolver only works for binary_classification"

        kernel = self.get_regularized_kernel(train_kernel).squeeze(0)
        train_targets = ((train_targets + 1) / 2).int() # convert back from {-1, 1} to {0, 1}

        self.logistic_model = LogisticRegressionCV(max_iter=10000, random_state=0)
        self.logistic_model.fit(kernel.cpu().numpy(), train_targets.cpu().numpy())

    def predict(self, eval_kernel, eval_targets, eval_logits=None, **unused_kwargs):
        assert self.logistic_model is not None, "Must call fit() before predict()"
        assert eval_kernel.size(0) == self.num_labels, "Number of labels in eval_kernel must match fit()"

        log_proba = self.logistic_model.predict_log_proba(eval_kernel.cpu().squeeze().t().numpy())
        log_proba = torch.tensor(log_proba, dtype=self.kernel_dtype, device=eval_kernel.device)

        eval_loss = self.loss(log_proba, eval_targets)

        eval_preds = (log_proba[:,1] - log_proba[:,0]).unsqueeze(1)

        return eval_loss, eval_preds

    def loss(self, preds, targets):
        targets = ((targets + 1) / 2).long() # convert back from {-1, 1} to {0, 1}
        return F.cross_entropy(preds, targets).item()



SOLVERS = {
    "lstsq": LstsqKernelSolver,
    "svr": SVRKernelSolver,
    "svc": SVCKernelSolver,
    "asym": AsymmetricLstsqKernelSolver,
    "logistic": LogisticKernelSolver
}
