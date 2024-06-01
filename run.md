# 指定 GPU 资源
srun -p q3090 --gres=gpu:rtx3090:2 --cpus-per-task=8 --pty --mail-type=ALL bash

# 安装依赖库
pip install torch==2.2.0
pip install accelerate==0.27.2

# 运行book
bash ./shell/instruct_7B.sh 1 1
bash ./shell/evaluate.sh 1 XXX_0_64

## 参数
instruction_model: alpaca-lora-7b是gpt3微调模型

## 脚本运行
bash ./shell/instruct_7B.sh 1 0
以上命令会输出到outputdir，输出checkpoint-120和checkpoint200，里面有adapter_config.json以及scheduler.pt、optimizer.pt学习率调度、优化器状态，模型权重文件training_args.bin

## 报错信息
```
safetensors/torch.py", line 308, in load_file
    with safe_open(filename, framework="pt", device=device) as f:
safetensors_rust.SafetensorError: Error while deserializing header: InvalidHeaderDeserialization
```
peft报错，重新安装peft解决
```
pip uninstall peft
pip install git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08
```

# 运行movie
bash ./shell/instruct_7B.sh 1 1
movie的运行结果
bash ./shell/evaluate.sh 1 movie2_1_64
```
"1": {
    "64": 0.6125449409347714
}
```

# 运行profile
bash ./shell/instruct_7B.sh 1 1
profile运行
eval非常耗时，修改eval_delay为50，在训练开始多少个epoch之后开始执行验证，修改eval_steps: 每隔多少个训练步执行一次验证为50

bash ./shell/evaluate.sh 1 finetuning_with_profile_1_64
# 报错
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 72.00 MiB. GPU 0 has a total capacty of 23.69 GiB of which 20.19 MiB is free. Including non-PyTorch memory, this process has 23.66 GiB memory in use. Of the allocated memory 21.84 GiB is allocated by PyTorch, and 1.52 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
```
超内存了，需要batch size改为8

只选前100查看结果，profile数据集测试准确率0.74。因为batchsize为8，11000个数据需要1375个迭代，需要约1个半小时。
每次跑验证集需要10分钟。

## 对比实验
验证微调的有效性，movie模型评估profile测试集
bash ./shell/evaluate.sh 1 movie2_1_64

## 输出response
增加以下代码
```
```
test的输出
bash ./shell/evaluate.sh 1 finetuning_with_profile_1_64


# 更换模型
llama3
bash ./shell/instruct_7B.sh 1 1
