import argparse
import os
import subprocess

from config_converter import BaseConfigConverter

ROOT = os.getcwd()

CONFIG_DICT = {
    'UNIQA': {
        'encoder_decoder_type': 'unilm',
        'pretrained_run_name': 'bert-base-uncased',
        'mlm_probability': 0.3,
        'attention_mask_type': 's2s_mask_nlq_20'
    },
    'T5': {
        'encoder_decoder_type': 't5',
        'pretrained_run_name': 't5-base',
        'attention_mask_type': 's2s'
    }
}

def add_fit_args(parser):
    parser.add_argument('--gpu_id', type=str, default='0', help="GPU ID (for the case ensemble_test, you should set 2 GPU IDs (e.g., '0,1'))")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ensemble_test', action='store_true')
    parser.add_argument('--dataset_dir', type=str, default='natural')
    parser.add_argument('--model', type=str, default='T5', help="T5 | UNIQA")
    
    parser.add_argument('--num_train_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=18)
    parser.add_argument('--eval_batch_size', type=int, default=16)

    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=1, help="the number of programs to generate")

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ensemble_seed', type=str, default='42,1,12,123,1234', help="(for ensemble_test) model seed for decoding")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = add_fit_args(argparse.ArgumentParser())
    import pdb; pdb.set_trace()
    init_config = CONFIG_DICT[args.model]
    init_config.update(vars(args))
    init_config['train_setting'] = 'finetune' if (args.test == False and args.ensemble_test == False) else 'decode'
    init_config['dataset'] = {'train': f'{args.dataset_dir}/train.json', 'eval': f'{args.dataset_dir}/dev.json', 'test': f'{args.dataset_dir}/test.json'}    
    
    if args.ensemble_test == False:
        SRC_PATH, TRAINING_CONFIG_LIST = BaseConfigConverter(init_config).get_subprocess_items()
    else:
        SRC_PATH, TRAINING_CONFIG_LIST = BaseConfigConverter(init_config).get_subprocess_items(ensemble=True)
        TRAINING_CONFIG_LIST.append("--{}={}".format('ensemble_seed', args.ensemble_seed))
    subprocess.run(['python', SRC_PATH] + TRAINING_CONFIG_LIST)
