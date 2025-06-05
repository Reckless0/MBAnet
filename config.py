import torch
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="arguments in processing data, building model and training")

    # data args
    # parser.add_argument("--data_dir", type=str, default="data/dataset", metavar="path",
    #                     help="path of dataset (default: 'data/dataset')")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        metavar="num_folds",
                        help="number of folds (default: 5)")
    parser.add_argument("--dataset",
                        type=str,
                        default=None, # /data16/linrun/BA_code_new/data/newest_m1_分割后+m2_整张_总_5fold_rename
                        )
    parser.add_argument("--test_dataset",
                        type=str,
                        default=None, # /data16/linrun/BA_code_new/data/external_video_top_5_percent_images_m12
                        )
    parser.add_argument('-nw','--num_workers',
                        type=int,
                        default=4, # 4 train 16 test
                        metavar='num_workers',
                        help='number of workers that pre-process data in parallel')
    parser.add_argument('--img_size',
                        type=int,
                        default=224, #224
                        metavar='image_size',
                        help='input image size after data augmentation')
    parser.add_argument('--use_meta', 
                        action='store_true')
    # parser.add_argument('--use_meta', 
    #                      type=bool,
    #                      default=False)
    parser.add_argument('--hospital_split', 
                        action='store_true')
    parser.add_argument('--modal',
                        type=str,
                        default=None)  # '12' '1' '2'

    parser.add_argument('--ext_video',  # external video dataset
                    action='store_true')
    parser.add_argument('--ext',        # external dataset
                    action='store_true')
    parser.add_argument('--int',        # internal dataset
                    action='store_true')

    # model args
    # - birectional
    parser.add_argument('--bi_directional',          # conv-layer proj feature map
                    action='store_true')
    parser.add_argument('--bi_directional_new_1',    # bilinear proj feature map
                    action='store_true')
    parser.add_argument('--bi_directional_new_2',    # maxpool proj feature map
                    action='store_true')
    

    # only use ONE
    parser.add_argument('--freeze_image_modality',  # freeze image-modality
                    action='store_true')
    parser.add_argument('--unfreeze_image_modality',   
                    action='store_true')

    parser.add_argument('--img_pretraing_path',
                    default=None,
                    type=str,
                    help='Image modality Pretrained model path (.pth).')
    
    ## - channel_attention
    parser.add_argument('--channel_attn',     
                    action='store_true')
    
    # - gate_attention
    parser.add_argument('--gate_attn',       
                    action='store_true')
    
    parser.add_argument('--device',
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        metavar='num_workers',
                        help='use GPU or CPU')
    parser.add_argument('--res_dir',
                        type=str,
                        default=None) # /data16/linrun/BA_results
    parser.add_argument('--eval_dir',
                        type=str,
                        default=None) # /data16/linrun/BA_results
    # parser.add_argument('--n_pretrain_classes',
    #                     default=2,
    #                     type=int,
    #                     help=('Number of classes of pretraining task.'
    #                           'When using --pretrain_path, this must be set.'))
    # parser.add_argument('--n_classes',
    #                     default=700,
    #                     type=int,
    #                     help=('Number of classes of pretraining task.'
    #                           'When using --pretrain_path, this must be set.'))



    # train args
    parser.add_argument('--use_trainning', 
                        action='store_true')
    parser.add_argument('-bs',"--batch_size",
                        type=int,
                        default=32, #32 train  128 for test
                        metavar="batch_size",
                        help="number of batch size")
    parser.add_argument('-e','--epochs',
                        type=int,
                        default=200,  # 200
                        metavar='num_epochs',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        metavar='lr',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha',
                        default=1.,
                        type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--patience',
                        default=20,
                        type=int,
                        help='patience for early stopping')
    parser.add_argument('--temp',
                        default=0.07,
                        type=float,
                        help='tempurature for SupConLoss (default: 0.07)')
    
    # eval args
    parser.add_argument("--weight_pr_ratio",
                        type=float,
                        default=1.0)
    parser.add_argument("--vote_ratio",
                        type=float,
                        default=0.05)


    # parser.add_argument('--checkpoint_dir',
    #                     default='save_checkpoint',
    #                     type=str,
    #                     help='checkpoints for log and models')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    # parser.add_argument('--model_depth',
    #                     default=50,
    #                     type=int,
    #                     help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    # parser.add_argument('--pretrain_path',
    #                     default="models/Pretrained_ResNet/pretrained_3dresnet/r3d101_K_200ep.pth",
    #                     type=Path,
    #                     help='Pretrained model path (.pth).')
    # parser.add_argument('--n_input_channels',
    #                     default=3,
    #                     type=int,
    #                     help='input channel of resnet')
    #
    # parser.add_argument('--conv1_t_size',
    #                     default=7,
    #                     type=int,
    #                     help='Kernel size in t dim of conv1.')
    # parser.add_argument('--conv1_t_stride',
    #                     default=1,
    #                     type=int,
    #                     help='Stride in t dim of conv1.')
    # parser.add_argument('--no_max_pool',
    #                     action='store_true',
    #                     help='If true, the max pooling after conv1 is removed.')
    # parser.add_argument('--resnet_shortcut',
    #                     default='B',
    #                     type=str,
    #                     help='Shortcut type of resnet (A | B)')
    # parser.add_argument(
    #     '--resnet_widen_factor',
    #     default=1.0,
    #     type=float,
    #     help='The number of feature maps of resnet is multiplied by this value')
    #
    # parser.add_argument(
    #     '--n_classes',
    #     default=2,
    #     type=int,
    #     help=
    #     'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    # )
    # parser.add_argument('--n_pretrain_classes',
    #                     default=700,
    #                     type=int,
    #                     help=('Number of classes of pretraining task.'
    #                           'When using --pretrain_path, this must be set.'))
    #
    # parser.add_argument(
    #     '--distributed',
    #     action='store_true',
    #     help='Use multi-processing distributed training to launch '
    #          'N processes per node, which has N GPUs.')

    return parser.parse_args()

    # opt = parser.parse_args([])