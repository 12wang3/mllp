import os
import argparse
import logging
import numpy as np
import torch

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from mllp.utils import read_csv, DBEncoder
from mllp.models import MLLP


DATA_DIR = 'dataset'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b):
    set_name = 'validation' if args.use_validation_set else 'training'

    fig = plt.figure(figsize=(16, 16))
    fig.suptitle('Dataset: {}'.format(args.data_set), fontsize=16)
    plt.subplot(3, 1, 1)
    loss_array = np.array(loss_log)

    plt.plot(loss_array, color='b', label='Total loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss during the training')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(np.array(accuracy), color='b', label='MLLP')
    plt.plot(np.array(accuracy_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(np.array(f1_score), color='b', label='MLLP')
    plt.plot(np.array(f1_score_b), color='g', label='CRS')

    plt.xlabel('epoch * 5')
    plt.ylabel('F1 Score Micro')
    plt.title('F1 Score (Macro) on the {} set'.format(set_name))
    plt.grid(True)
    plt.legend()

    plt.savefig(args.plot_file)


def experiment(args):
    dataset = args.data_set
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[args.ith_kfold]
    X_train_df = X_df.iloc[train_index]
    y_train_df = y_df.iloc[train_index]
    X_test_df = X_df.iloc[test_index]
    y_test_df = y_df.iloc[test_index]

    logging.info('Discretizing and binarizing data. Please wait ...')
    db_enc = DBEncoder(f_df, discrete=True)
    db_enc.fit(X_df, y_df)
    X_fname = db_enc.X_fname
    y_fname = db_enc.y_fname
    X_train, y_train = db_enc.transform(X_train_df, y_train_df)
    X_test, y_test = db_enc.transform(X_test_df, y_test_df)
    logging.info('Data discretization and binarization are done.')

    if args.use_validation_set:
        # Use 20% of the training set as the validation set.
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, validation_index = next(kf.split(X_train))
        X_validation = X_train[validation_index]
        y_validation = y_train[validation_index]
        X_train = X_train[train_index]
        y_train = y_train[train_index]
    else:
        X_validation = None
        y_validation = None

    net_structure = [len(X_fname)] + list(map(int, args.structure.split('_'))) + [len(y_fname)]
    net = MLLP(net_structure,
               device=device,
               random_binarization_rate=args.random_binarization_rate,
               use_not=args.use_not,
               log_file=args.log)
    net.to(device)

    loss_log, accuracy, accuracy_b, f1_score, f1_score_b = net.train(
        X_train,
        y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        epoch=args.epoch,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epoch=args.lr_decay_epoch,
        weight_decay=args.weight_decay)

    plot_loss(args, loss_log, accuracy, accuracy_b, f1_score, f1_score_b)

    acc, acc_b, f1, f1_b = net.test(X_test, y_test, need_transform=True)
    logging.info('=' * 60)
    logging.info('Test:\n\tAccuracy of MLLP Model: {}\n\tAccuracy of CRS  Model: {}'.format(acc, acc_b))
    logging.info('Test:\n\tF1 Score of MLLP Model: {}\n\tF1 Score of CRS  Model: {}'.format(f1, f1_b))
    logging.info('=' * 60)

    with open(args.crs_file, 'w') as f:
        net.concept_rule_set_print(X_train, X_fname, y_fname, f)
    torch.save(net.state_dict(), args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                        help='Set the data set for training. All the data sets in the dataset folder are available.')
    parser.add_argument('-k', '--kfold', type=int, default=5, help='Set the k of K-Folds cross-validation.')
    parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th validation, 0 <= ki < k.')
    parser.add_argument('--use_validation_set', action="store_true",
                        help='Use the validation set for parameters tuning.')
    parser.add_argument('-e', '--epoch', type=int, default=401, help='Set the total epoch.')
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help='Set the batch size.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help='Set the initial learning rate.')
    parser.add_argument('-lrdr', '--lr_decay_rate', type=float, default=0.75, help='Set the learning rate decay rate.')
    parser.add_argument('-lrde', '--lr_decay_epoch', type=int, default=100, help='Set the learning rate decay epoch.')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='Set the weight decay (L2 penalty).')
    parser.add_argument('-p', '--random_binarization_rate', type=float, default=0.0,
                        help='Set the rate of random binarization. It is important for CRS extractions from deep MLLPs.')
    parser.add_argument('--use_not', action="store_true",
                        help='Use the NOT (~) operator in logical rules. '
                             'It will enhance model capability but make the CRS more complex.')
    parser.add_argument('-s', '--structure', type=str, default='64',
                        help='Set the structure of network. Only the number of nodes in middle layers are needed. '
                             'E.g., 64, 64_32_16. The total number of middle layers should be odd.')

    args = parser.parse_args()
    args.folder_name = '{}_k{}_ki{}_useValidationSet{}_e{}_bs{}_lr{}_lrdr{}_lrde{}_wd{}_p{}_useNOT{}'.format(
        args.data_set, args.kfold, args.ith_kfold, args.use_validation_set, args.epoch, args.batch_size,
        args.learning_rate, args.lr_decay_rate, args.lr_decay_epoch, args.weight_decay,
        args.random_binarization_rate, args.use_not)

    if not os.path.exists('log_folder'):
        os.mkdir('log_folder')
    args.folder_name = args.folder_name + '_L' + args.structure
    args.folder_path = os.path.join('log_folder', args.folder_name)
    if not os.path.exists(args.folder_path):
        os.mkdir(args.folder_path)
    args.model = os.path.join(args.folder_path, 'model.pth')
    args.crs_file = os.path.join(args.folder_path, 'crs.txt')
    args.plot_file = os.path.join(args.folder_path, 'plot_file.pdf')
    args.log = os.path.join(args.folder_path, 'log.txt')
    logging.basicConfig(level=logging.INFO, filename=args.log, filemode='w', format='[%(levelname)s] - %(message)s')
    # for arg in vars(args):
    #     print(arg, getattr(args, arg))
    experiment(args)
