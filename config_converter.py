import os

# Transformation to exectuable configuration
class ConfigConverter:
    def __init__(self, init_config):
        os.environ["CUDA_VISIBLE_DEVICES"] = init_config['gpu_id']
        self.cur_dir = os.getcwd()
        
        self.ex_config = {}
        self.setup_data_args(init_config=init_config)
        self.setup_model_args(init_config=init_config)
        self.setup_training_args(init_config=init_config)
        
        self.setup_run_name(init_config=init_config)
        self.setup_output_dir(init_config=init_config)
    
    def setup_data_args(self, init_config):
        train_file_name = init_config['dataset']['train']
        eval_file_name = init_config['dataset']['eval']
        test_file_name = init_config['dataset']['test']
        
        # check .json
        train_dataset = train_file_name if '.json' in train_file_name else os.path.join(train_file_name, 'train.json')
        eval_dataset = eval_file_name if '.json' in eval_file_name else os.path.join(eval_file_name, 'dev.json')
        test_dataset = test_file_name if '.json' in test_file_name else os.path.join(test_file_name, 'test.json')
        
        self.ex_config['train_data_file'] = os.path.join(*[self.cur_dir, f'data', train_dataset])
        self.ex_config['eval_data_file'] = os.path.join(*[self.cur_dir, f'data', eval_dataset])
        self.ex_config['test_data_file'] = os.path.join(*[self.cur_dir, f'data', test_dataset])
        
        if init_config['encoder_decoder_type'] == 'unilm':
            self.ex_config['txt_len'] = 100
            self.ex_config['trace_len'] = 280

        if 'block_size' in init_config:
            self.ex_config['block_size'] = init_config['block_size']
        self.ex_config['dataloader_num_workers'] = 256


    def setup_model_args(self, init_config):
        if init_config['encoder_decoder_type'] == 't5':
            self.ex_config['encoder_decoder_type'] = init_config['encoder_decoder_type']
            self.ex_config['model_name_or_path'] = init_config['pretrained_run_name']
            
        elif init_config['encoder_decoder_type'] == 'unilm':
            self.ex_config['encoder_decoder_type'] = init_config['encoder_decoder_type']
            self.ex_config['model_name_or_path'] = init_config['pretrained_run_name']
        else:
            raise ValueError("possible encoder_decoder_type: unilm / t5")

    def setup_training_args(self, init_config):
        self.ex_config['train_setting'] = init_config['train_setting']
        
        self.ex_config['num_train_epochs'] = init_config['num_train_epochs']
        self.ex_config['train_batch_size'] = init_config['train_batch_size']
        self.ex_config['eval_batch_size'] = init_config['eval_batch_size']
        self.ex_config['learning_rate'] = init_config['learning_rate']
        
        if 'mim_probability' in init_config:
            self.ex_config['mlm_probability'] = init_config['mlm_probability']
        self.ex_config['attention_mask_type'] = init_config['attention_mask_type']  # bi, bar, s2s, s2s_mask_nlq_{.2d}
        self.ex_config['seed'] = init_config['seed']
        
        if init_config['train_setting'] in ['pretrain', 'finetune']:
            self.ex_config['do_train']   = True
            self.ex_config['do_eval']    = True
            self.ex_config['do_predict'] = False
            self.ex_config['mlm']        = True
            
            # assertion the attention mask
            if init_config['train_setting'] == 'pretrain':
                assert init_config['attention_mask_type'] in ['bi', 'bar']
            elif init_config['train_setting'] == 'finetune':
                possible_mask_type1 = ['s2s', 's2s_mask_nlq_05', 's2s_mask_nlq_10', 's2s_mask_nlq_20', 's2s_mask_nlq_30']
                possible_mask_type2 = ['s2s_only_mask_nlq_05', 's2s_only_mask_nlq_10', 's2s_only_mask_nlq_20']
                assert init_config['attention_mask_type'] in possible_mask_type1 + possible_mask_type2
    
        elif init_config['train_setting'] in ['decode']:
            self.ex_config['do_train']   = False
            self.ex_config['do_eval']    = False
            self.ex_config['do_predict'] = True
            self.ex_config['mlm']        = False
            self.ex_config['recover']    = True

            # decode specific setting
            self.ex_config['num_samples'] = init_config['num_samples'] if 'num_samples' in init_config else 1
            self.ex_config['beam_size'] = init_config['beam_size'] if 'beam_size' in init_config else 1
            if 'top_p' in init_config: self.ex_config['top_p'] = init_config['top_p']
            
            # reset attention mask if mis-typed
            if init_config['attention_mask_type'] in ['s2s', 's2s_mask_nlq_05', 's2s_mask_nlq_10', 's2s_mask_nlq_20', 's2s_mask_nlq_30']:
                self.ex_config['attention_mask_type'] = 's2s'
            
        else:
            raise ValueError("possible train_setting: pretrain, finetune, decode")

    def setup_run_name(self, init_config):
        pretrained_model_name = init_config['pretrained_run_name']
        
        train_file_name = init_config['dataset']['train'].split('.')[0] if '.json' in init_config['dataset']['train'] else init_config['dataset']['train']
        
        ne = init_config['num_train_epochs']
        lr = init_config['learning_rate']
        mlm_prob = f"_mlm_{init_config['mlm_probability']}" if 'mlm_probability' in init_config else ''
        attn_mask = init_config['attention_mask_type']
        seed = init_config['seed']

        # for logging
        pretrain_name = '-'.join(init_config['pretrained_run_name'].split('/')[-6:])+'/' if 'saved/models' in init_config['pretrained_run_name'] else ''
        self.RUN_NAME = f'{pretrain_name}{train_file_name}/{pretrained_model_name}/ne{ne}_lr{lr}{mlm_prob}_{attn_mask}_{seed}'
        self.ex_config['run_name'] = self.RUN_NAME

    def setup_output_dir(self, init_config):
        self.ex_config['output_dir'] = os.path.join(self.cur_dir, f"saved/models/pretrained_models/{self.RUN_NAME}")
            
    def get_subprocess_items(self, ensemble=False):
        ex_config = self.ex_config
        if ensemble:
            SRC_PATH = f"model/ensemble_test.py"
        else:
            SRC_PATH = f"trainer/run_trainer.py"
        TRAINING_CONFIG_LIST = list()
        for (k,v) in list(ex_config.items()):
            if (isinstance(v, bool)):
                if v:
                    TRAINING_CONFIG_LIST.append("--{}".format(k))
            else:
                TRAINING_CONFIG_LIST.append("--{}={}".format(k,v))
        return SRC_PATH, TRAINING_CONFIG_LIST


''' use the model trained on BASE'''
class BaseConfigConverter(ConfigConverter):
    def __init__(self, init_config):
        super().__init__(init_config=init_config)