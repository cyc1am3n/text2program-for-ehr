# Base pkgs
import os
import csv
import math
from typing import Optional
import logging
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.optim import Adadelta, Adagrad, Adam, AdamW
import pytorch_lightning as pl

# Transformers pkgs
from transformers import get_linear_schedule_with_warmup
from transformers import (
    CONFIG_MAPPING,
    AutoConfig
)
# For beam search
from utils.beam_utils import BeamSearchScorer
from transformers.generation_stopping_criteria import validate_stopping_criteria
from transformers.generation_logits_process import LogitsProcessorList
from transformers.generation_stopping_criteria import StoppingCriteriaList
from transformers.generation_utils import BeamSearchEncoderDecoderOutput


# Custom pkgs
from .metrics import get_accuracy
from .transformer import Text2TraceT5Model
from .unilm import Text2TraceBertForMaskedLM, Text2TraceBertForGeneration


logger = logging.getLogger(__name__)


''' finetune/decode '''
class Text2TraceForTransformerModel(pl.LightningModule):
    def __init__(self, model_args, training_args, data_args, tokenizer):
        super().__init__()

        self.eval_sets = {0:"tot", 1:"txt", 1:"trace"}

        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.tokenizer = tokenizer
        
        # Load Config and Encoder-Decoder Model
        def _load_config_and_model(model_args):
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            model = Text2TraceT5Model(
                model_args.model_name_or_path,
                config=config,
                tokenizer=self.tokenizer,
            )
            return config, model

        self.config, self.model = _load_config_and_model(model_args=self.model_args)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x).logits

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, encoder_input_ids=None, encoder_attention_mask=None, past=None, **kwargs
    ):
        def _prepare_decoder_inputs_for_generation(input_ids, attention_mask=None, past=None):
            if attention_mask is None:
                attention_mask = (input_ids != self.config.pad_token_id).type(torch.int64).to(input_ids.device)
            return_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'past_key_values': past
            }
            return return_dict

        decoder_inputs = _prepare_decoder_inputs_for_generation(input_ids, attention_mask, past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "input_ids": encoder_input_ids,
            "attention_mask": encoder_attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            # "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            # "use_cache": use_cache,
        }
        return input_dict
    
    def _reorder_cache(self, past, beam_idx):
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    def beam_search(self,
                    input_ids: torch.LongTensor,
                    decoder_start_token_id: int,
                    attention_mask: torch.LongTensor,
                    num_beams: int,
                    num_return_sequences: int,
                    length_penalty: Optional[float] = None,
                    early_stopping: Optional[bool] = None,
                    logits_processor: Optional[LogitsProcessorList] = None,
                    stopping_criteria: Optional[StoppingCriteriaList] = None,
                    max_length: Optional[int] = None,
                    pad_token_id: Optional[int] = None,
                    eos_token_id: Optional[int] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    output_scores: Optional[bool] = None,
                    return_dict_in_generate: Optional[bool] = None,
                    synced_gpus: Optional[bool] = None,
                    **model_kwargs,):

        # init values
        batch_size = input_ids.shape[0]
        
        length_penalty = length_penalty if length_penalty is not None else 1.0 # 1.0 means no penalty
        early_stopping = early_stopping if early_stopping is not None else False

        if num_return_sequences > num_beams:
            raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=input_ids.device,
            length_penalty=length_penalty,
            do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences,
        )

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        max_length = max_length if max_length is not None else self.config.max_length
        validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        # interleave with 'num_beams'
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(input_ids.device)
        )
        model_kwargs['encoder_input_ids'] = input_ids.index_select(0, expanded_return_idx)
        if attention_mask is not None:
            model_kwargs['encoder_attention_mask'] = attention_mask.index_select(0, expanded_return_idx)

        model_kwargs['decoder_start_token_id'] = decoder_start_token_id
        input_ids = torch.ones((model_kwargs['encoder_input_ids'].shape[0], 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        batch_beam_size, cur_len = input_ids.shape

        output_logits = None
        assert (
            num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        
        breakpoint()
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]                    # (batch_size * num_beams, vocab_size)
            
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `F.log_softmax` operation.
            # next_token_logits = self.adjust_logits_during_generation(
            #     next_token_logits, cur_len=cur_len, max_length=max_length
            # )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                output_logits,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            next_token_logits = next_token_logits.unsqueeze(1)
            if output_logits is None:
                output_logits = next_token_logits[beam_idx, :, :]
            else:
                output_logits = torch.cat([output_logits[beam_idx, :, :], next_token_logits[beam_idx, :, :]], dim=1)

            cur_len = cur_len + 1

            if "past_key_values" in outputs:
                model_kwargs["past"] = outputs.past_key_values
            
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

            if stopping_criteria(input_ids, scores):
                break
        sequence_outputs = beam_scorer.finalize(
            input_ids, output_logits, beam_scores, next_tokens, next_indices, vocab_size=vocab_size, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=sequence_outputs["sequence_logits"],#output_logits,#scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return sequence_outputs["sequences"]
        

    def training_step(self, batch, batch_idx):
        # memorize total loss
        output = self.model(**batch)
        loss = output.loss
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metrics = {}
        val_tot_acc, val_txt_acc, val_trace_acc = 0.0, 0.0, 0.0

        # 1. compute the validation loss
        outputs = self.model(**batch)
        #breakpoint()
        if isinstance(outputs, dict):
            loss, logits = outputs["loss"], outputs["logits"]
        else:
            loss, logits = outputs[0], outputs[1]

        # 2. compute the accuracy
        if batch["labels"] is not None:
            label = batch["labels"].data
            label = torch.cat([label[:, 1:], label.new(label.shape[0], 1).fill_(-100)], axis=1)
            val_trace_acc = get_accuracy(logit=logits.data, label=label)

        if batch["text_labels"] is not None:
            text_labels = batch["text_labels"]
            text_logits = outputs["text_logits"]
            val_txt_acc = get_accuracy(logit=text_logits, label=text_labels)
        
        # If fine-tuning, 3. decode and check lf accuracy
        if self.training_args.train_setting == "finetune":
            eval_result = self.model.decode_for_evaluation(**batch)

        val_metrics = {
            'val_loss': loss,
            'val_trace_acc': val_trace_acc,
            'val_txt_acc': val_txt_acc,
        }
        
        val_metrics_trace = {'val_ex_cnt': eval_result}
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {**val_metrics, **val_metrics_trace}
    
    def validation_epoch_end(self, val_step_outputs):
        assert type(val_step_outputs) == list
        assert type(val_step_outputs[0]) == dict
        
        val_tot_ex_cnt = 0

        for out in val_step_outputs:
            val_tot_ex_cnt += out['val_ex_cnt']

        val_dataloader = self.val_dataloader()
        val_dataset_size = len(val_dataloader.dataset)
    
        self.log('val_ex_cnt', val_tot_ex_cnt, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_ex_acc', val_tot_ex_cnt / val_dataset_size, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    
        
    def test_step(self, batch, batch_idx):
        return None

    def test_step_end(self, test_step_outputs):
       return None
   
    def test_epoch_end(self, test_epoch_outputs):
        # NOTE: We skip the test step procedure !
        '''Called at the end of a test epoch with the output of all test steps.'''
        from .evaluation import EvalForMimicProgram
        
        # define modules depend on database
        # if self.data_args.database == 'multi_line' or self.data_args.database == 'single_line': #self.data_args.database == 'mimicprogram':
        eval_module = EvalForMimicProgram
        self.eval_module = eval_module(
                data_args=self.data_args,
                training_args=self.training_args,
                model_args=self.model_args,
                tokenizer=self.tokenizer,
            )
        # run evaluate for val file (it could be another test file)
        # self._test_epoch_end_depend_on_state(state='val')
        
        # run evaluate for test file
        self._test_epoch_end_depend_on_state(state='test')        
    
    def _test_epoch_end_depend_on_state(self, state='test'):
        from utils.eval_utils import gather_evaluation_outputs, write_decode_output_file
        '''run evaluation lop for test dataset '''
        
        # make directory if not exists
        save_dir = os.path.join(os.getcwd(), f"saved/decode_outputs/{self.training_args.run_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        # define model
        model = self#.model
        
        # define data_loader, data_file, flag_test depend on state(val/test)
        data_loader = self.test_dataloader()
        data_file_path = self.data_args.test_data_file
        
        # decide the file path
        suffix = f'_beam_{self.training_args.beam_size}' if self.training_args.beam_size != 1 else ''
        suffix = suffix + f'_top_p_{self.training_args.top_p}' if self.training_args.top_p is not None else suffix
        suffix = suffix + f'_top_k_{self.training_args.top_k}' if self.training_args.top_k is not None else suffix
        suffix = suffix + f'_n_{self.training_args.num_samples}' if self.training_args.num_samples != 1 else suffix
        results_fname = data_file_path.split("/data/")[1].replace('/', '_').replace('.json', f'{suffix}.csv')
        results_fpath = os.path.join(save_dir, results_fname)
        
        # init the logging metirc
        final_ex_cnt = 0
        
        # When target path has already the file, load the evaluation result file.
        flag_no_load = False  # For recover flag
        if os.path.isfile(results_fpath):
            results = []
            with open(results_fpath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    results.append(row)
            final_ex_cnt = sum([row['ex_flag'] == 'True' for row in results])
            if self.training_args.recover:
                if 'recover_pred' in results[0]:
                    final_recover_ex_cnt = sum([row['recover_ex_flag'] == 'True' for row in results])
                else:
                    flag_no_load = True

        # When target path does not have the file, do evaluation.
        if not(os.path.isfile(results_fpath)) or flag_no_load:
            results = []
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                batch = self.transfer_batch_to_device(batch)
                result = self.eval_module.evaluate_step(model=model, batch=batch, batch_idx=batch_idx)
                results.append(result)
            results = gather_evaluation_outputs(results, self.training_args.recover)
            final_ex_cnt = sum(results['ex_flag'])
            if self.training_args.recover:
                final_recover_ex_cnt = sum(results['recover_ex_flag'])
            write_decode_output_file(save_file_path=results_fpath, save_file=results, recover=self.training_args.recover)
        
        # Write logging
        self.log(f'{state}_ex', final_ex_cnt)
        
        if self.training_args.recover:
            self.log(f'{state}_recover_ex', final_recover_ex_cnt)

    def configure_optimizers(self):
        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
        )

        # steps
        num_update_steps_per_epoch = len(self.train_dataloader()) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.training_args.max_steps > 0:
            max_steps = self.training_args.max_steps
            num_train_epochs = self.training_args.max_steps // num_update_steps_per_epoch + int(
                self.training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)

        # lr_scheduler
        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=max_steps,
            ),
            'interval': 'step',
        }
                    
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_callbacks(self):
        '''Automatically monitor and logs learning rate for learning rate schedulers during training.'''
        lr_monitor = pl.callbacks.LearningRateMonitor(
            logging_interval="step",
        )      

        if self.training_args.train_setting == 'pretrain':
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=self.training_args.output_dir,
                filename='{epoch}-{val_loss:.2f}',
                monitor='val_loss',
                save_top_k=1,
                mode='min',
            )
            return [lr_monitor, checkpoint]
        elif self.training_args.train_setting == 'finetune':
            monitor_metric = "val_ex_acc"
            save_filename = '{epoch}-{val_loss:.2f}-{val_ex_acc:.2f}'
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=self.training_args.output_dir,
                filename=save_filename,
                monitor=monitor_metric,
                save_top_k=1,
                verbose=True,
                mode='max',
            )
            early_stop = pl.callbacks.EarlyStopping(
                monitor=monitor_metric,
                min_delta=0.0,
                patience=3, # 5
                verbose=True,
                mode="max",
            )
            return [lr_monitor, checkpoint, early_stop]
        else:
            return None

''' pretrain/finetune '''
class Text2TraceForUnilmModel(pl.LightningModule):
    def __init__(self, model_args, training_args, data_args, tokenizer):
        super().__init__()
        self.eval_sets = {0:"tot", 1:"txt", 1:"trace"}
        
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.tokenizer = tokenizer
        
        config, model = self._load_config_and_model(model_args=self.model_args)
        logger.info(f"Resize token embeddings of model by {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
        self.model = model
        self.save_hyperparameters()
        if self.training_args.tie_word_embeddings:
            model.tie_weights()

    def forward(self, x):
        '''
        Same as torch.nn.Module.forward(), however in Lightning you want this to define the operations you want to use for prediction (i.e.: on a server or as a feature extractor).
        Normally you’d call self() from your training_step() method. This makes it easy to write a complex system for training with the outputs you’d want in a prediction setting.
        You may also find the auto_move_data() decorator useful when using the module outside Lightning in a production setting.
        '''
        # self(x)
        # NOTE: not used for right now
        return self.model(x).logits
    
    def training_step(self, batch, batch_idx):
        # memorize total loss
        # output format -> Text2SQLMaskedLMOutput
        outputs = self.model(**batch)
        
        if isinstance(outputs, dict):
            loss, logits = outputs["loss"], outputs["logits"]
        else:
            loss, logits = outputs[0], outputs[2]
            
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True) # followed by LightningModule Hook
        return loss

    def validation_step(self, batch, batch_idx):
        '''
        Operates on a single batch of data from the validation set. In this step you’d might generate examples or calculate anything of interest like accuracy.
        If you pass in multiple val dataloaders, validation_step() will have an additional argument. (dataloader_idx tells you which dataset this is.)
        '''
        metrics = {}
        val_tot_acc, val_txt_acc, val_trace_acc = 0.0, 0.0, 0.0
        
        # 1. compute the validation loss
        # loss is tensor. The Checkpoint Callback is monitoring 'checkpoint_on'
        outputs = self.model(**batch)
        if isinstance(outputs, dict):
            loss, logits = outputs["loss"], outputs["logits"]
        else:
            loss, logits = outputs[0], outputs[2]
        
        # 2. compute the accuracy
        if batch["labels"] is not None:
            val_tot_acc = get_accuracy(logit=logits.data, label=batch["labels"].data)
        if batch["text_labels"] is not None:
            val_txt_acc = get_accuracy(logit=logits.data, label=batch["text_labels"].data)
        if batch["trace_labels"] is not None:
            val_trace_acc = get_accuracy(logit=logits.data, label=batch["trace_labels"].data)

        # If fine-tuning, 3. decode and check lf accuracy
        if self.training_args.train_setting == "finetune":
            eval_result = self.model.decode_for_evaluation(**batch)
            
        val_metrics = {
            'val_loss': loss,
            'val_tot_acc': val_tot_acc,
            'val_txt_acc': val_txt_acc,
            'val_trace_acc': val_trace_acc,
        }
        val_metrics_trace = {'val_ex_cnt' : eval_result}
        
        self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True) # followed by LightningModule Hook
            
        return {**val_metrics, **val_metrics_trace}
    
    def validation_epoch_end(self, val_step_outputs):
        ''' Called at the end of the validation epoch with the outputs of all validation steps. '''
        assert type(val_step_outputs) == list
        assert type(val_step_outputs[0]) == dict
        
        val_tot_ex_cnt = 0
        
        for out in val_step_outputs:
            val_tot_ex_cnt += out['val_ex_cnt']
        
        val_dataloader = self.val_dataloader()
        val_dataset_size = len(val_dataloader.dataset)
        
        self.log('val_ex_cnt', val_tot_ex_cnt, on_step=False, on_epoch=True, prog_bar=False, logger=True) # followed by LightningModule Hook
        self.log('val_ex_acc', val_tot_ex_cnt/val_dataset_size, on_step=False, on_epoch=True, prog_bar=False, logger=True) # followed by LightningModule Hook

        
    def test_step(self, batch, batch_idx):
        return None

    def test_step_end(self, test_step_outputs):
       return None

    def test_epoch_end(self, test_epoch_outputs):
        # NOTE: We skip the test step procedure !
        '''Called at the end of a test epoch with the output of all test steps.'''
        from .evaluation import EvalForMimicProgram
        
        # define modules depend on database
        # if self.data_args.database == 'multi_line' or self.data_args.database == 'single_line': #self.data_args.database == 'mimicprogram':
        eval_module = EvalForMimicProgram
        self.eval_module = eval_module(
                data_args=self.data_args,
                training_args=self.training_args,
                model_args=self.model_args,
                tokenizer=self.tokenizer,
            )
        # run evaluate for val file (it could be another test file)
        # self._test_epoch_end_depend_on_state(state='val')
        
        # run evaluate for test file
        self._test_epoch_end_depend_on_state(state='test')        
    
    def _test_epoch_end_depend_on_state(self, state='test'):
        from utils.eval_utils import gather_evaluation_outputs, write_decode_output_file
        '''run evaluation lop for test dataset '''
        
        # make directory if not exists
        save_dir = os.path.join(os.getcwd(), f"saved/decode_outputs/{self.training_args.run_name}")
        os.makedirs(save_dir, exist_ok=True)
        
        # define model
        model = self.model
        
        # define data_loader, data_file, flag_test depend on state(val/test)
        data_loader = self.test_dataloader()
        data_file_path = self.data_args.test_data_file
        
        # decide the file path
        suffix = f'_beam_{self.training_args.beam_size}' if self.training_args.beam_size != 1 else ''
        suffix = suffix + f'_top_p_{self.training_args.top_p}' if self.training_args.top_p is not None else suffix
        suffix = suffix + f'_top_k_{self.training_args.top_k}' if self.training_args.top_k is not None else suffix
        suffix = suffix + f'_n_{self.training_args.num_samples}' if self.training_args.num_samples != 1 else suffix
        results_fname = data_file_path.split("/data/")[1].replace('/', '_').replace('.json', f'{suffix}.csv')
        results_fpath = os.path.join(save_dir, results_fname)
        
        # init the logging metirc
        final_ex_cnt = 0
        
        # When target path has already the file, load the evaluation result file.
        flag_no_load = False  # For recover flag
        if os.path.isfile(results_fpath):
            results = []
            with open(results_fpath, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    results.append(row)
            #breakpoint()
            assert len(results) == len(data_loader.dataset)
            final_ex_cnt = sum([row['ex_flag'] == 'True' for row in results])
            if self.training_args.recover:
                if 'recover_pred' in results[0]:
                    final_recover_ex_cnt = sum([row['recover_ex_flag'] == 'True' for row in results])
                else:
                    flag_no_load = True

        # When target path does not have the file, do evaluation.
        if not(os.path.isfile(results_fpath)) or flag_no_load:
            results = []
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                batch = self.transfer_batch_to_device(batch)
                result = self.eval_module.evaluate_step(model=model, batch=batch, batch_idx=batch_idx)
                results.append(result)
            results = gather_evaluation_outputs(results, self.training_args.recover)
            final_ex_cnt = sum(results['ex_flag'])
            if self.training_args.recover:
                final_recover_ex_cnt = sum(results['recover_ex_flag'])
            write_decode_output_file(save_file_path=results_fpath, save_file=results, recover=self.training_args.recover)
        
        # Write logging
        self.log(f'{state}_ex', final_ex_cnt)
        
        if self.training_args.recover:
            self.log(f'{state}_recover_ex', final_recover_ex_cnt)

    def configure_optimizers(self):
        # optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.training_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_epsilon,
        )

        # steps
        num_update_steps_per_epoch = len(self.train_dataloader()) // self.training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        if self.training_args.max_steps > 0:
            max_steps = self.training_args.max_steps
            num_train_epochs = self.training_args.max_steps // num_update_steps_per_epoch + int(
                self.training_args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            max_steps = math.ceil(self.training_args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(self.training_args.num_train_epochs)

        # lr_scheduler
        lr_scheduler = {
            'scheduler': get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.training_args.warmup_steps,
                num_training_steps=max_steps,
            ),# ReduceLROnPlateau(optimizer),
            'interval': 'step',
            # 'monitor': 'train_loss',
            # 'frequency': 10,
        }
                    
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    
    def configure_callbacks(self):
        
        '''Automatically monitor and logs learning rate for learning rate schedulers during training.'''
        lr_monitor = pl.callbacks.LearningRateMonitor(
            logging_interval="step",
        )
        
        if self.training_args.train_setting == 'pretrain':
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=self.training_args.output_dir,
                filename='{epoch}-{val_loss:.2f}',
                monitor='val_loss',
                save_top_k=1,
                mode='min',
            )
            return [lr_monitor, checkpoint]
        elif self.training_args.train_setting == 'finetune':
            save_filename = '{epoch}-{val_loss:.2f}-{val_ex_acc:.2f}'
            monitor_metric = "val_ex_acc"
            checkpoint = pl.callbacks.ModelCheckpoint(
                dirpath=self.training_args.output_dir,
                filename = save_filename,
                monitor=monitor_metric,
                save_top_k=1,
                verbose=True,
                mode='max',
            )
            early_stop = pl.callbacks.EarlyStopping(
                monitor=monitor_metric,
                min_delta=0.0,
                patience=3, # 5
                verbose=True,
                mode="max",
                # check_finite=True,
                # stopping_threshold=0.9
            )
            return [lr_monitor, checkpoint, early_stop]
        else:
            return None
    
    def _load_config_and_model(self, model_args):
        
        # config part
        if model_args.config_name:
            config = AutoConfig.from_pretrained(model_args.config_name)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")
            
        # model part - finetune
        if self.training_args.train_setting in ['finetune', 'pretrain']:
            if model_args.model_name_or_path:
                model = Text2TraceBertForMaskedLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    tokenizer=self.tokenizer,
                )
            else:
                logger.info("Training new model from scratch")
                model = Text2TraceBertForMaskedLM.from_config(config=config)
        # model part - decode
        elif self.training_args.train_setting == 'decode':
            model = Text2TraceBertForGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                tokenizer=self.tokenizer,
            )
        model.txt_len = self.data_args.txt_len
        return config, model