import sys
import logging
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from collections import defaultdict

from mllp.utils import UnionFind

THRESHOLD = 0.5


class RandomlyBinarize(torch.autograd.Function):
    """Implement the forward and backward propagation of the random binarization operation."""

    @staticmethod
    def forward(ctx, W, M):
        W = W.clone()
        W[M] = torch.where(W[M] > THRESHOLD, torch.ones_like(W[M]), torch.zeros_like(W[M]))
        ctx.save_for_backward(M.type(torch.float))
        return W

    @staticmethod
    def backward(ctx, grad_output):
        M, = ctx.saved_tensors
        grad_input = grad_output * (1.0 - M)
        return grad_input, None


class RandomBinarizationLayer(nn.Module):
    """Implement the Random Binarization (RB) method."""

    def __init__(self, shape, probability):
        super(RandomBinarizationLayer, self).__init__()
        self.shape = shape
        self.probability = probability
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.M = torch.rand(self.shape, device=self.device) < self.probability

    def forward(self, W):
        return RandomlyBinarize.apply(W, self.M)

    def refresh(self):
        self.M = torch.rand(self.shape, device=self.device) < self.probability


class ConjunctionLayer(nn.Module):
    """The conjunction layer is used to learn the rules."""

    def __init__(self, n, input_dim, random_binarization_rate, use_not=False):
        super(ConjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2

        # Initialization: set Wi close to 0 to avoid gradient being too small.
        self.W = nn.Parameter(0.1 * torch.rand(self.n, self.input_dim))
        self.node_activation_cnt = None
        self.randomly_binarize_layer = RandomBinarizationLayer(self.W.shape, random_binarization_rate)

    def forward(self, x, randomly_binarize=False):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        W = self.randomly_binarize_layer(self.W) if randomly_binarize else self.W
        # This formula will generate a batch_size*L1*L2 tensor,
        # and it could cost lots of memory if batch_size is very large.
        return torch.prod((1 - (1 - x)[:, :, None] * W.t()[None, :, :]), dim=1)

    def binarized_forward(self, x):
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            Wb = torch.where(self.W > THRESHOLD, torch.ones_like(self.W), torch.zeros_like(self.W)).type(torch.int)
            return torch.prod((1 - (1 - x)[:, :, None] * Wb.t()[None, :, :]), dim=1)


class DisjunctionLayer(nn.Module):
    """The disjunction layer is used to learn the rule sets."""

    def __init__(self, n, input_dim, random_binarization_rate, use_not=False):
        super(DisjunctionLayer, self).__init__()
        self.n = n
        self.use_not = use_not
        self.input_dim = input_dim if not use_not else input_dim * 2

        # Initialization: set Wi close to 0 to avoid gradient being too small.
        self.W = nn.Parameter(0.1 * torch.rand(self.n, self.input_dim))
        self.node_activation_cnt = None
        self.randomly_binarize_layer = RandomBinarizationLayer(self.W.shape, random_binarization_rate)

    def forward(self, x, randomly_binarize=False):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        W = self.randomly_binarize_layer(self.W) if randomly_binarize else self.W
        # This formula will generate a batch_size*L1*L2 tensor,
        # and it could cost lots of memory if batch_size is very large.
        return 1 - torch.prod(1 - x[:, :, None] * W.t()[None, :, :], dim=1)

    def binarized_forward(self, x):
        with torch.no_grad():
            if self.use_not:
                x = torch.cat((x, 1 - x), dim=1)
            x = x.type(torch.int)
            Wb = torch.where(self.W > THRESHOLD, torch.ones_like(self.W), torch.zeros_like(self.W)).type(torch.int)
            return 1 - torch.prod(1 - x[:, :, None] * Wb.t()[None, :, :], dim=1)


class MLLP(nn.Module):
    """The Multilayer Logical Perceptron (MLLP) used for Concept Rule Sets (CRS) learning.

    For more information, please read our paper: Transparent Classification with Multilayer Logical Perceptrons and
    Random Binarization."""

    def __init__(self, dim_list, device, random_binarization_rate=0.75, use_not=False, log_file=None):
        """

        Parameters
        ----------
        dim_list : list
            A list specifies the number of nodes (neurons) of all the layers in MLLP from bottom to top. dim_list[0]
            should be the dimensionality of the input data and dim_list[1] should be the number of class labels.
        device : torch.device
            Run on which device.
        random_binarization_rate : float
            The rate of the random binarization in the Random Binarizatoin (RB) method. RB method is important for CRS
            extractions from deep MLLPs.
        use_not : bool
            Whether use the NOT (~) operator in logical rules.
        log_file : str
            The path of the log file. If log_file is None, use sys.stdout as the output stream.
        """

        super(MLLP, self).__init__()

        log_format = '[%(levelname)s] - %(message)s'
        if log_file is None:
            logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format)
        else:
            logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w', format=log_format)

        self.dim_list = dim_list
        self.device = device
        self.use_not = use_not
        self.enc = None
        self.conj = []
        self.disj = []
        for i in range(0, len(dim_list) - 2, 2):
            conj = ConjunctionLayer(dim_list[i + 1], dim_list[i], random_binarization_rate, use_not=use_not)
            disj = DisjunctionLayer(dim_list[i + 2], dim_list[i + 1], random_binarization_rate, use_not=False)
            self.add_module('conj{}'.format(i), conj)
            self.add_module('disj{}'.format(i), disj)
            self.conj.append(conj)
            self.disj.append(disj)

    def forward(self, x, randomly_binarize=False):
        for conj, disj in zip(self.conj, self.disj):
            x = conj(x, randomly_binarize=randomly_binarize)
            x = disj(x, randomly_binarize=randomly_binarize)
        return x

    def binarized_forward(self, x):
        """Equivalent to using the extracted Concept Rule Sets."""
        with torch.no_grad():
            for conj, disj in zip(self.conj, self.disj):
                x = conj.binarized_forward(x)
                x = disj.binarized_forward(x)
        return x

    def clip(self):
        """Clip the weights into the range [0, 1]."""
        for param in self.parameters():
            param.data.clamp_(0, 1)

    def randomly_binarize_layer_refresh(self):
        """Change the set of weights to be binarized."""
        for conj, disj in zip(self.conj, self.disj):
            conj.randomly_binarize_layer.refresh()
            disj.randomly_binarize_layer.refresh()

    def binarized(self):
        for param in self.parameters():
            param.data = torch.where(param.data > THRESHOLD, torch.ones_like(param.data), torch.zeros_like(param.data))

    def data_transform(self, X, y):
        X = X.astype(np.float32)
        if y is None:
            return torch.tensor(X)
        y = y.astype(np.float32)
        logging.debug('{}'.format(y.shape))
        logging.debug('{}'.format(y[:20]))
        return torch.tensor(X), torch.tensor(y)  # Do not put all the data in GPU at once.

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_rate=0.9, lr_decay_epoch=7):
        """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs."""
        lr = init_lr * (lr_decay_rate ** (epoch // lr_decay_epoch))
        if epoch % lr_decay_epoch == 0:
            logging.info('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train(self, X=None, y=None, X_validation=None, y_validation=None, data_loader=None, epoch=50, lr=0.01, lr_decay_epoch=100,
              lr_decay_rate=0.75, batch_size=64, weight_decay=0.0):
        """

        Parameters
        ----------
        X : numpy.ndarray, shape = [n_samples, n_features]
            The training input instances. All the values should be 0 or 1.
        y : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels. All the values should be 0 or 1.
        X_validation : numpy.ndarray, shape = [n_samples, n_features]
            The input instances of validation set. The format of X_validation is the same as X.
            if X_validation is None, use the training set (X) for validation.
        y_validation : numpy.ndarray, shape = [n_samples, n_classes]
            The class labels of validation set. The format of y_validation is the same as y.
            if y_validation is None, use the training set (y) for validation.
        epoch : int
            The total number of epochs during the training.
        lr : float
            The initial learning rate.
        lr_decay_epoch : int
            Decay learning rate every lr_decay_epoch epochs.
        lr_decay_rate : float
            Decay learning rate by a factor of lr_decay_rate.
        batch_size : int
            The batch size for training.
        weight_decay : float
            The weight decay (L2 penalty).

        Returns
        -------
        loss_log : list
            Training loss of MLLP during the training.
        accuracy : list
            Accuracy of MLLP on the validation set during the training.
        accuracy_b : list
            Accuracy of CRS on the validation set during the training.
        f1_score : list
            F1 score (Macro) of MLLP on the validation set during the training.
        f1_score_b : list
            F1 score (Macro) of CRS on the validation set during the training.

        """

        if (X is None or y is None) and data_loader is None:
            raise Exception("Both data set and data loader are unavailable.")
        if data_loader is None:
            X, y = self.data_transform(X, y)
            if X_validation is not None and y_validation is not None:
                X_validation, y_validation = self.data_transform(X_validation, y_validation)
            data_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

        loss_log = []
        accuracy = []
        accuracy_b = []
        f1_score = []
        f1_score_b = []

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        for epo in range(epoch):
            optimizer = self.exp_lr_scheduler(optimizer, epo, init_lr=lr, lr_decay_rate=lr_decay_rate,
                                              lr_decay_epoch=lr_decay_epoch)
            running_loss = 0.0
            cnt = 0
            for X, y in data_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()  # Zero the gradient buffers.
                y_pred = self.forward(X, randomly_binarize=True)
                loss = criterion(y_pred, y)
                running_loss += loss.item()
                loss.backward()
                if epo % 100 == 0 and cnt == 0:
                    for param in self.parameters():
                        logging.debug('{}'.format(param.grad))
                    cnt += 1
                optimizer.step()
                self.clip()
            logging.info('epoch: {}, loss: {}'.format(epo, running_loss))
            loss_log.append(running_loss)
            # Change the set of weights to be binarized every epoch.
            self.randomly_binarize_layer_refresh()

            # Test the validation set or training set every 5 epochs.
            if epo % 5 == 0:
                if X_validation is not None and y_validation is not None:
                    acc, acc_b, f1, f1_b = self.test(X_validation, y_validation, False)
                    set_name = 'Validation'
                else:
                    acc, acc_b, f1, f1_b = self.test(X, y, False)
                    set_name = 'Training'
                logging.info('-' * 60)
                logging.info('On {} Set:\n\tAccuracy of MLLP Model: {}'
                             '\n\tAccuracy of CRS  Model: {}'.format(set_name, acc, acc_b))
                logging.info('On {} Set:\n\tF1 Score of MLLP Model: {}'
                             '\n\tF1 Score of CRS  Model: {}'.format(set_name, f1, f1_b))
                logging.info('-' * 60)
                accuracy.append(acc)
                accuracy_b.append(acc_b)
                f1_score.append(f1)
                f1_score_b.append(f1_b)
        return loss_log, accuracy, accuracy_b, f1_score, f1_score_b

    def test(self, X, y, need_transform=True):
        if need_transform:
            X, y = self.data_transform(X, y)
        with torch.no_grad():
            X = X.to(self.device)
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)

            y = y.cpu().numpy().astype(np.int)
            y = np.argmax(y, axis=1)
            data_num = y.shape[0]
            slice_step = data_num // 40 if data_num >= 40 else 1
            logging.debug('{} {}'.format(y.shape, y[:: slice_step]))

            # Test the model batch by batch.
            # Test the MLLP.
            y_pred_list = []
            for X, in test_loader:
                y_pred_list.append(self.forward(X))
            y_pred = torch.cat(y_pred_list)
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            logging.debug('{} {}'.format(y_pred.shape, y_pred[:: slice_step]))

            # Test the CRS.
            y_pred_b_list = []
            for X, in test_loader:
                y_pred_b_list.append(self.binarized_forward(X))
            y_pred_b = torch.cat(y_pred_b_list)
            y_pred_b = y_pred_b.cpu().numpy()
            logging.debug('y_pred_b: {} {}'.format(y_pred_b.shape, y_pred_b[:: (slice_step * 2)]))
            y_pred_b = np.argmax(y_pred_b, axis=1)
            logging.debug('{} {}'.format(y_pred_b.shape, y_pred_b[:: slice_step]))

            accuracy = metrics.accuracy_score(y, y_pred)
            accuracy_b = metrics.accuracy_score(y, y_pred_b)

            f1_score = metrics.f1_score(y, y_pred, average='macro')
            f1_score_b = metrics.f1_score(y, y_pred_b, average='macro')
        return accuracy, accuracy_b, f1_score, f1_score_b

    def detect_dead_node(self, X, need_transform=True):
        if need_transform:
            X = self.data_transform(X, None)
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=False)
            for conj, disj in zip(self.conj, self.disj):
                conj.node_activation_cnt = torch.zeros(conj.n, dtype=torch.long, device=self.device)
                disj.node_activation_cnt = torch.zeros(disj.n, dtype=torch.long, device=self.device)

            # Test the model batch by batch.
            for x, in test_loader:
                x = x.to(self.device)
                for conj, disj in zip(self.conj, self.disj):
                    x = conj.binarized_forward(x)
                    conj.node_activation_cnt += torch.sum(x, dim=0)
                    x = disj.binarized_forward(x)
                    disj.node_activation_cnt += torch.sum(x, dim=0)

    def get_rules(self, X=None):
        """Extract rules from parameters of MLLP."""
        # If X is not None, detect the dead nodes using X.
        if X is not None:
            self.detect_dead_node(X)
            activation_cnt_list = [np.sum(np.concatenate([X, 1 - X], axis=1) if self.use_not else X, axis=0)]
            for conj, disj in zip(self.conj, self.disj):
                activation_cnt_list.append(conj.node_activation_cnt.cpu().numpy())
                activation_cnt_list.append(disj.node_activation_cnt.cpu().numpy())
        else:
            activation_cnt_list = None

        # Get the rules from the top layer to the bottom layer.
        param_list = list(self.parameters())
        n_param = len(param_list)
        mark = {}
        rules_list = []
        for i in reversed(range(n_param)):
            param = param_list[i]
            W = param.cpu().detach().numpy()
            rules = defaultdict(list)
            num = self.dim_list[i]
            for k, row in enumerate(W):
                if i != n_param - 1 and ((i, k) not in mark):
                    continue
                if X is not None and activation_cnt_list[i + 1][k] < 1:
                    continue
                found = False
                for j, wj in enumerate(row):
                    if X is not None and activation_cnt_list[i][j % num] < 1:
                        continue
                    if wj > THRESHOLD:
                        rules[k].append(j)
                        mark[(i - 1, j % num)] = 1
                        found = True
                if not found:
                    rules[k] = []
            rules_list.append(rules)
        return rules_list

    def eliminate_redundant_rules(self, rules_list):
        """Eliminate redundant rules to simplify the extracted CRS."""
        rules_list = copy.deepcopy(rules_list)
        for i in reversed(range(len(rules_list))):
            # Eliminate the redundant part of each rule from bottom to top.
            if i != len(rules_list) - 1:
                num = self.dim_list[len(self.dim_list) - i - 2]
                for k, v in rules_list[i].items():
                    mark = {}
                    new_rule = []
                    for j1 in range(len(v)):
                        if j1 in mark:
                            continue
                        for j2 in range(j1 + 1, len(v)):
                            if j2 in mark:
                                continue
                            if j1 // num != j2 // num:
                                continue
                            s1 = set(rules_list[i + 1][v[j1 % num]])
                            s2 = set(rules_list[i + 1][v[j2 % num]])
                            if s1.issuperset(s2):
                                mark[j1] = 1
                                break
                            elif s1.issubset(s2):
                                mark[j2] = 1
                        if j1 not in mark:
                            new_rule.append(v[j1])
                    rules_list[i][k] = sorted(list(set(new_rule)))

            # Merge the identical nodes.
            union_find = UnionFind(rules_list[i].keys())
            kv_list = list(rules_list[i].items())
            n_kv = len(kv_list)
            if i > 0:
                for j1 in range(n_kv):
                    k1, v1 = kv_list[j1]
                    for j2 in range(j1 + 1, n_kv):
                        k2, v2 = kv_list[j2]
                        if v1 == v2:
                            union_find.union(k1, k2)
                # Update the upper layer.
                for k, v in rules_list[i - 1].items():
                    for j in range(len(v)):
                        v[j] = union_find.find(v[j])
                    rules_list[i - 1][k] = sorted(list(set(v)))
        # Get the final simplified rules.
        new_rules_list = []
        mark = {}
        for i in range(len(rules_list)):
            num = self.dim_list[len(self.dim_list) - i - 2]
            rules = defaultdict(list)
            for k, v in rules_list[i].items():
                if i != 0 and ((i, k) not in mark):
                    continue
                for j in v:
                    mark[(i + 1, j % num)] = 1
                    rules[k].append(j)
            new_rules_list.append(rules)
        return new_rules_list

    def get_name(self, i, j, X_fname=None, y_fname=None):
        nl = len(self.dim_list)
        num = self.dim_list[nl - i - 1]
        if j >= num:
            j -= num
            prefix = '~'
        else:
            prefix = ' '
        if X_fname is not None and i == nl - 1:
            name = X_fname[j]
        elif y_fname is not None and i == 0:
            name = y_fname[j]
        else:
            name = '{}{},{}'.format('s' if i % 2 == 0 else 'r', (nl - 2 - i) // 2 + 1, j)
        name = prefix + name
        return name

    def concept_rule_set_print(self, X=None, X_fname=None, y_fname=None, file=sys.stdout, eliminate_redundancy=True):
        """Print the Concept Rule Sets extracted from the trained Multilayer Logical Perceptron."""
        if eliminate_redundancy:
            rules_list = self.eliminate_redundant_rules(self.get_rules(X))
        else:
            rules_list = self.get_rules(X)
        for i in range(0, len(rules_list), 2):
            rules_str = defaultdict(list)
            for k, v in rules_list[i + 1].items():
                for j in v:
                    rules_str[k].append(self.get_name(i + 2, j, X_fname=X_fname, y_fname=y_fname))
            rule_sets = defaultdict(list)
            num = self.dim_list[len(self.dim_list) - i - 2]
            for k, v in rules_list[i].items():
                for j in v:
                    if j >= num:
                        jn = j - num
                        prefix = '~'
                    else:
                        prefix = ' '
                        jn = j
                    rule_sets[self.get_name(i, k, X_fname=X_fname, y_fname=y_fname)].append(
                        '{:>10}:\t{}{}'.format(self.get_name(i + 1, j, X_fname=X_fname, y_fname=y_fname), prefix,
                                               rules_str[jn]))
            print('-' * 90, file=file)
            for k, v in rule_sets.items():
                print('{}:'.format(k), file=file)
                for r in v:
                    print('\t', r, file=file)
