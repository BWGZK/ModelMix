import argparse


def config(name='MyoPS'):

    parser = argparse.ArgumentParser(description=name)

    parser.add_argument('--path', type=str, default='/data/kzhang99/ModelMix/MyoPS_clean_v2/', help="data path")
    parser.add_argument('--load_path', type=str, default='/data/ModelMix_results/myops/', help="load path")
    parser.add_argument('--predict_mode', type=str, default='single', help="predict mode: single or multiple")
    parser.add_argument('--test_path', type=str, default='test', help="test path")

    # parameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--dim', type=int, default=192, help="dimension of 2D image")
    parser.add_argument('--feature_dim', type=int, default=1024, help="dimensions of feature")
    # training setting
    parser.add_argument('--lr', type=float, default=1e-4, help="starting learning rate")
    parser.add_argument('--threshold', type=float, default=0.40, help="the minimum dice to save or predict model")

    parser.add_argument('--start_epoch', type=int, default=0, help="flag to indicate the start epoch")
    parser.add_argument('--end_epoch', type=int, default=500, help="flag to indicate the final epoch")

    args = parser.parse_args()

    return args
    