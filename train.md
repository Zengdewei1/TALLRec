# 训练日志

```
(graphgpt) h3619835@gpu-3090-402:~/TALLRec$ bash ./shell/instruct_7B.sh 1 0
1, 0
lr: 1e-4, dropout: 0.05 , seed: 0, sample: 64

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda118.so
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /userhome/34/h3619835/miniconda3/envs/graphgpt did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/data/baokq/miniconda3/envs/alpaca_lora/lib')}
  warn(msg)
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: /data/baokq/miniconda3/envs/alpaca_lora/lib/ did not contain ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] as expected! Searching further paths...
  warn(msg)
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/run/user/13756/vscode-git-6dcd573c6b.sock')}
  warn(msg)
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/run/user/13756/vscode-ipc-d2723de6-e49a-4482-ba37-ee665827e9a5.sock')}
  warn(msg)
CUDA_SETUP: WARNING! libcudart.so not found in any environmental path. Searching in backup paths...
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: Found duplicate ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] files: {PosixPath('/usr/local/cuda/lib64/libcudart.so'), PosixPath('/usr/local/cuda/lib64/libcudart.so.11.0')}.. We'll flip a coin and try one of these, in order to fail forward.
Either way, this might cause trouble in the future:
If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.6
CUDA SETUP: Detected CUDA version 118
CUDA SETUP: Loading binary /userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda118.so...
[2024-05-28 10:41:55,808] [INFO] [real_accelerator.py:133:get_accelerator] Setting ds_accelerator to cuda (auto detect)
Training Alpaca-LoRA model with params:
base_model: baffo32/decapoda-research-llama-7B-hf
train_data_path: ./data/movie/train.json
val_data_path: ./data/movie/valid.json
sample: 64
seed: 0
output_dir: XXX_0_64
batch_size: 128
micro_batch_size: 32
num_epochs: 200
learning_rate: 0.0001
cutoff_len: 512
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules: ['q_proj', 'v_proj']
train_on_inputs: True
group_by_length: True
wandb_project: 
wandb_run_name: 
wandb_watch: 
wandb_log_model: 
resume_from_checkpoint: tloen/alpaca-lora-7b

config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 428/428 [00:00<00:00, 25.7kB/s]
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
pytorch_model.bin.index.json: 100%|██████████████████████████████████████████████████████████████████████████████████| 25.5k/25.5k [00:00<00:00, 1.73MB/s]
pytorch_model-00001-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:24<00:00, 16.7MB/s]
pytorch_model-00002-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:10<00:00, 39.7MB/s]
pytorch_model-00003-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:11<00:00, 36.3MB/s]
pytorch_model-00004-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:36<00:00, 11.1MB/s]
pytorch_model-00005-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:10<00:00, 39.3MB/s]
pytorch_model-00006-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:10<00:00, 38.3MB/s]
pytorch_model-00007-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:25<00:00, 15.9MB/s]
pytorch_model-00008-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:24<00:00, 16.5MB/s]
pytorch_model-00009-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.0MB/s]
pytorch_model-00010-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:10<00:00, 39.7MB/s]
pytorch_model-00011-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.2MB/s]
pytorch_model-00012-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:26<00:00, 15.4MB/s]
pytorch_model-00013-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 16.9MB/s]
pytorch_model-00014-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:35<00:00, 11.3MB/s]
pytorch_model-00015-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.1MB/s]
pytorch_model-00016-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.0MB/s]
pytorch_model-00017-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.0MB/s]
pytorch_model-00018-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.1MB/s]
pytorch_model-00019-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.1MB/s]
pytorch_model-00020-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:35<00:00, 11.3MB/s]
pytorch_model-00021-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:34<00:00, 11.7MB/s]
pytorch_model-00022-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.2MB/s]
pytorch_model-00023-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:09<00:00, 40.9MB/s]
pytorch_model-00024-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:09<00:00, 41.0MB/s]
pytorch_model-00025-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.2MB/s]
pytorch_model-00026-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.0MB/s]
pytorch_model-00027-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:23<00:00, 17.3MB/s]
pytorch_model-00028-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:13<00:00, 30.4MB/s]
pytorch_model-00029-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [02:14<00:00, 3.00MB/s]
pytorch_model-00030-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:24<00:00, 16.8MB/s]
pytorch_model-00031-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:24<00:00, 16.8MB/s]
pytorch_model-00032-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 405M/405M [00:24<00:00, 16.8MB/s]
pytorch_model-00033-of-00033.bin: 100%|████████████████████████████████████████████████████████████████████████████████| 524M/524M [00:13<00:00, 40.1MB/s]
Downloading shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 33/33 [14:47<00:00, 26.88s/it]
Loading checkpoint shards:   0%|                                                                                                   | 0/33 [00:00<?, ?it/s]/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 33/33 [02:40<00:00,  4.85s/it]
generation_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 124/124 [00:00<00:00, 8.37kB/s]
tokenizer_config.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 142/142 [00:00<00:00, 57.9kB/s]
tokenizer.model: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 500k/500k [00:00<00:00, 3.51MB/s]
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2.00/2.00 [00:00<00:00, 1.04kB/s]
You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama.LlamaTokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/peft/utils/other.py:102: FutureWarning: prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.
  warnings.warn(
Downloading and preparing dataset json/default to /userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51...
Downloading data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 2304.56it/s]
Extracting data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 14.00it/s]
Dataset json downloaded and prepared to /userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51. Subsequent calls will reuse this data.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.36it/s]
Downloading and preparing dataset json/default to /userhome/34/h3619835/.cache/huggingface/datasets/json/default-8f061eccffcfddce/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51...
Downloading data files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1979.38it/s]
Extracting data files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 15.97it/s]
Dataset json downloaded and prepared to /userhome/34/h3619835/.cache/huggingface/datasets/json/default-8f061eccffcfddce/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51. Subsequent calls will reuse this data.
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 72.27it/s]
Checkpoint tloen/alpaca-lora-7b/adapter_model.bin not found
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
Traceback (most recent call last):
  File "finetune_rec.py", line 325, in <module>
    fire.Fire(train)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "finetune_rec.py", line 245, in train
    trainer = transformers.Trainer(
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/trainer.py", line 373, in __init__
    self.create_accelerator_and_postprocess()
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/trainer.py", line 4252, in create_accelerator_and_postprocess
    self.accelerator = Accelerator(
TypeError: __init__() got an unexpected keyword argument 'use_seedable_sampler'
```


# 2
```
Found cached dataset json (/userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 76.89it/s]
Found cached dataset json (/userhome/34/h3619835/.cache/huggingface/datasets/json/default-8f061eccffcfddce/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51)
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 183.10it/s]
Checkpoint tloen/alpaca-lora-7b/adapter_model.bin not found
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
Loading cached shuffled indices for dataset at /userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/cache-441f1dabd96b225f.arrow
Loading cached shuffled indices for dataset at /userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/cache-d1da15a069405e3c.arrow
Loading cached processed dataset at /userhome/34/h3619835/.cache/huggingface/datasets/json/default-36d2f41eb4083dc2/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/cache-74052f73ed15a18f.arrow
Loading cached processed dataset at /userhome/34/h3619835/.cache/huggingface/datasets/json/default-8f061eccffcfddce/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/cache-0817556e1da7b487.arrow
Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the --report_to flag to control the integrations used for logging result (for instance --report_to none).
  0%|                                                                                                                             | 0/200 [00:00<?, ?it/s]/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
  2%|██▉                                                                                                                | 5/200 [01:40<1:03:41, 19.60s/it]
```



```
{'loss': 0.1962, 'grad_norm': 0.07550367712974548, 'learning_rate': 4.4444444444444447e-05, 'epoch': 120.0}
{'eval_loss': 0.4486449956893921, 'eval_auc': 0.5942227021454346, 'eval_runtime': 64.5861, 'eval_samples_per_second': 15.483, 'eval_steps_per_second': 1.935, 'epoch': 120.0}
 60%|████████████████████████████████████████████████████████████████████████████████████████████▍                                                             | 120/200 [51:39<26:49, 20.12s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.1903, 'grad_norm': 0.07868025451898575, 'learning_rate': 4e-05, 'epoch': 128.0}
{'eval_loss': 0.4524643123149872, 'eval_auc': 0.5789540340212234, 'eval_runtime': 64.6063, 'eval_samples_per_second': 15.478, 'eval_steps_per_second': 1.935, 'epoch': 130.0}
 65%|████████████████████████████████████████████████████████████████████████████████████████████████████                                                      | 130/200 [55:57<23:32, 20.17s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")




 67%|███████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                  | 134/200 [57:15<28:50, 26.22s/it]{'loss': 0.1849, 'grad_norm': 0.05556967481970787, 'learning_rate': 3.555555555555556e-05, 'epoch': 136.0}
{'eval_loss': 0.45707154273986816, 'eval_auc': 0.5783763486731013, 'eval_runtime': 64.8767, 'eval_samples_per_second': 15.414, 'eval_steps_per_second': 1.927, 'epoch': 140.0}
 70%|██████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                             | 140/200 [1:00:16<20:10, 20.17s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.18, 'grad_norm': 0.05724487453699112, 'learning_rate': 3.111111111111111e-05, 'epoch': 144.0}
{'eval_loss': 0.4623327851295471, 'eval_auc': 0.5647607493912086, 'eval_runtime': 64.5195, 'eval_samples_per_second': 15.499, 'eval_steps_per_second': 1.937, 'epoch': 150.0}
 75%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                      | 150/200 [1:04:34<16:48, 20.17s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.1753, 'grad_norm': 0.052243225276470184, 'learning_rate': 2.6666666666666667e-05, 'epoch': 152.0}
{'loss': 0.1711, 'grad_norm': 0.06260429322719574, 'learning_rate': 2.2222222222222223e-05, 'epoch': 160.0}
{'eval_loss': 0.46946635842323303, 'eval_auc': 0.5783296894719067, 'eval_runtime': 64.5736, 'eval_samples_per_second': 15.486, 'eval_steps_per_second': 1.936, 'epoch': 160.0}
 80%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                              | 160/200 [1:08:53<13:26, 20.16s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.1676, 'grad_norm': 0.05562916770577431, 'learning_rate': 1.777777777777778e-05, 'epoch': 168.0}
{'eval_loss': 0.4754062592983246, 'eval_auc': 0.5691022769690183, 'eval_runtime': 64.6322, 'eval_samples_per_second': 15.472, 'eval_steps_per_second': 1.934, 'epoch': 170.0}
 85%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                      | 170/200 [1:13:11<10:04, 20.15s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.1645, 'grad_norm': 0.07247364521026611, 'learning_rate': 1.3333333333333333e-05, 'epoch': 176.0}
{'eval_loss': 0.4787144064903259, 'eval_auc': 0.5816291615563732, 'eval_runtime': 64.8502, 'eval_samples_per_second': 15.42, 'eval_steps_per_second': 1.928, 'epoch': 180.0}
 90%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊               | 180/200 [1:17:30<06:43, 20.16s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.162, 'grad_norm': 0.06776496767997742, 'learning_rate': 8.88888888888889e-06, 'epoch': 184.0}
{'eval_loss': 0.48015472292900085, 'eval_auc': 0.5635209477594696, 'eval_runtime': 64.8492, 'eval_samples_per_second': 15.42, 'eval_steps_per_second': 1.928, 'epoch': 190.0}
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍       | 190/200 [1:21:48<03:21, 20.16s/it/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/bitsandbytes/autograd/_functions.py:321: UserWarning: MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization
  warnings.warn(f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")
{'loss': 0.1603, 'grad_norm': 0.0784970223903656, 'learning_rate': 4.444444444444445e-06, 'epoch': 192.0}
{'loss': 0.1593, 'grad_norm': 0.07658522576093674, 'learning_rate': 0.0, 'epoch': 200.0}
{'eval_loss': 0.48290562629699707, 'eval_auc': 0.5749080147176452, 'eval_runtime': 64.8337, 'eval_samples_per_second': 15.424, 'eval_steps_per_second': 1.928, 'epoch': 200.0}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [1:26:07<00:00, 20.16s/itTraceback (most recent call last):
  File "finetune_rec.py", line 325, in <module>
    fire.Fire(train)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "finetune_rec.py", line 292, in train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/trainer.py", line 1780, in train
    return inner_training_loop(
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/trainer.py", line 2241, in _inner_training_loop
    self._load_best_model()
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/trainer.py", line 2494, in _load_best_model
    model.load_adapter(self.state.best_model_checkpoint, model.active_adapter)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/peft/peft_model.py", line 554, in load_adapter
    adapters_weights = safe_load_file(filename, device="cuda" if torch.cuda.is_available() else "cpu")
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/safetensors/torch.py", line 308, in load_file
    with safe_open(filename, framework="pt", device=device) as f:
safetensors_rust.SafetensorError: Error while deserializing header: InvalidHeaderDeserialization

```

```
Traceback (most recent call last):
  File "evaluate.py", line 225, in <module>
    fire.Fire(main)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/fire/core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "evaluate.py", line 54, in main
    seed = temp_list[-2]
IndexError: list index out of range

```