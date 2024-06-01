pip3 install "fschat[model_worker,webui]==0.1.10"

srun -p q3090 --gres=gpu:rtx3090:2 --cpus-per-task=8 --pty --mail-type=ALL bash

3.2 instruction tuning
子命令 python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20001     graphgpt/train/train_mem.py
sh scripts/tune_script/graphgpt_stage1.sh


pip install fastchat
报错，因为配置路径不对
```
huggingface_hub.utils._validators.HFValidationError: Repo id must use alphanumeric chars or '-', '_', '.', '--' and '..' are forbidden, '-' and '.' cannot start or end the name, max length is 96: '../vicuna-7b-v1.5-16k'.
```
```
  File "/userhome/34/h3619835/GraphGPT/graphgpt/train/train_graph.py", line 763, in train
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/hf_argparser.py", line 338, in parse_args_into_dataclasses
You are using a model of type llama to instantiate a model of type GraphLlama. This is not supported for all configurations of models and can yield errors.
```


```
    param_persistence_threshold = get_config_default(DeepSpeedZeroConfig, "param_persistence_threshold")
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/deepspeed/runtime/config_utils.py", line 115, in get_config_default
        assert not config.__fields__.get(assert not config.__fields__.get(

AttributeErrorAttributeError: : 'FieldInfo' object has no attribute 'required''FieldInfo' object has no attribute 'required'

    assert not config.__fields__.get(
AttributeError: 'FieldInfo' object has no attribute 'required'
```

```
File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/torch/cuda/init.py", line 326, in set_device
cached = self.fget(obj)
File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/training_args.py", line 1684, in _setup_devices
torch.cuda.set_device(device)
File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/torch/cuda/init.py", line 326, in set_device
torch._C._cuda_setDevice(device)
RuntimeError: CUDA error: invalid device ordinal


If you get `CUDA error: invalid device function` errors, the above might be the cause and the solution is to make sure only one ['libcudart.so', 'libcudart.so.11.0', 'libcudart.so.12.0'] in the paths that we search based on your env.
  warn(msg)
```


```
Failures:
[1]:
  time      : 2024-04-12_14:58:30
  host      : gpu-3090-201.gpufarm1.cs.hku.hk
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 386091)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2024-04-12_14:58:30
  host      : gpu-3090-201.gpufarm1.cs.hku.hk
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 386090)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
```

```
Traceback (most recent call last):
  File "graphgpt/train/train_mem.py", line 17, in <module>
    train()
  File "/userhome/34/h3619835/GraphGPT/graphgpt/train/train_graph.py", line 803, in train
    model.config.pretrain_graph_model_path = model.config.pretrain_graph_model_path + model_args.graph_tower
  File "/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/transformers/configuration_utils.py", line 260, in __getattribute__
    return super().__getattribute__(key)
AttributeError: 'GraphLlamaConfig' object has no attribute 'pretrain_graph_model_path'
```
解决https://github.com/HKUDS/GraphGPT/issues/7
```

```
必须是pretra_gnn=clip_gt_arxiv，"pretrain_graph_model_path": "/userhome/34/h3619835/GraphGPT/"

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 172.00 MiB (GPU 1; 23.69 GiB total capacity; 23.21 GiB already allocated; 124.19 MiB free; 23.21 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
解决 lightning


train_light

scripts/tune_script_light/graphgpt_stage1_lightning.sh

pip install "fschat[model_worker,webui]==0.2.22"

warning，可以使用srun去使用slurm
```
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/lightning/fabric/plugins/environments/slurm.py:204: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python graphgpt/train/train_light.py --model_name_or_path . ...
```
可以降低GPU浮点精度
```
You are using a CUDA device ('NVIDIA GeForce RTX 3090') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
```
进程数量不够
```
/userhome/34/h3619835/miniconda3/envs/graphgpt/lib/python3.8/site-packages/torch/utils/data/dataloader.py:557: UserWarning: This DataLoader will create 16 worker processes in total. Our suggested max number of worker in current system is 8, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
```