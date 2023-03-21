import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="MixGCF")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="amazon",
                        help="Choose a dataset:[amazon-book,yelp2018]")
    parser.add_argument("--data_path", nargs="?", default="./data/", help="Input data path.")
    parser.add_argument('--name',        default='testrun',                  help='Set run name for saving/restoring models')
    # ===== train ===== #
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train_ratio')
    parser.add_argument("--gnn", nargs="?", default="ngcf",
                        help="Choose a recommender:[lightgcn, ngcf]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument("--mess_dropout", type=bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    # ===== train ===== #
    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    parser.add_argument("--cuda", type=bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=2, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='Output sizes of every layer')
    parser.add_argument("--context_hops", type=int, default=3, help="hop")
    parser.add_argument("--eval_earlystop", type=str, default='recall@20', help="evaluation metrics")
    # ===== model hyper parameters ===== #
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')
    parser.add_argument('--temperature_2', type=float, default=1.0, help='temperature_2')
    parser.add_argument('--temperature_3', type=float, default=1.0, help='temperature_2')
    parser.add_argument('--u_norm', dest='u_norm', action='store_true', help='whether to normalize the user embedding')
    parser.add_argument('--i_norm', dest='i_norm', action='store_true', help='whether to normalize the item embedding')
    # ===== save and log ===== #
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")
    parser.add_argument('--logdir',          dest='log_dir',         default='./log/',               help='Log directory')
    parser.add_argument('--config',          dest='config_dir',      default='./config/',            help='Config directory')
    parser.add_argument('--restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
    parser.add_argument('--loss_fn', type=str, default="PairwiseLogisticLoss", help="loss for training")
    parser.add_argument('--sampling_method', type=str, default="uniform", help="sampling_method")
    parser.add_argument('--generate_mode', type=str, default="normal", help="generate_mode")
    parser.add_argument('--tau_mode', type=str, default="static", help='method for calculate tau')
    parser.add_argument('--cnt_lr', type=int, default=10, help='epochs for warm-up')
    
    return parser.parse_args()
