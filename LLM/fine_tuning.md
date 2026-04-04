# 微调

## 场景

使用特定领域的数据训练，提升在这个领域内的回答质量

核心：学会指令遵循，把知识按照要求表达出来

六大场景：

- 回答风格调整（lora微调）
- 问答对记忆
- 领域知识灌注
- 代码/数学能力增强
- function calling能力增强
- agent能力增强

和预训练的区别

- completion only，只有答案部分计算loss



## NEF Tune （Noisy Embeddings Fine-Tuning）

带噪音的微调。通过在训练时向 `input_embeddings` 注入均匀分布的随机噪声，来增强模型的泛化能力，防止过拟合。研究表明，这种方法在微调（Fine-tuning）阶段能显著提升指令遵循模型的表现。

**transformer包中已通过超参支持**

```python
# NEFTune 噪声缩放系数，通常取值 5, 10 或 15
neftune_noise_alpha = 10

for i in range(epoch):
    for inputs, loss_mask in data_loader:
        # 获取输入的 Token ID
        input_ids = inputs.pop("input_ids")
        
        # 将 Token ID 转换为嵌入向量 (Embedding)
        input_embeddings = model.base_model.model.model.embed_tokens(input_ids)
        
        # 计算总维度：序列长度 * 隐藏层维度
        dims = torch.tensor(input_embeddings.size(1) * input_embeddings.size(2))
        
        # 计算噪声的标准差/幅度缩放因子
        mag_norm = neftune_noise_alpha / torch.sqrt(dims)
        
        # 生成均匀分布的噪声并叠加到原始嵌入向量上
        input_embeddings = input_embeddings + torch.zeros_like(input_embeddings).uniform_(-mag_norm, mag_norm)
        
        # 将处理后的嵌入向量存入 inputs 字典，后续传给模型
        inputs["inputs_embeds"] = input_embeddings
```



## 六种微调方法

- 全量微调：与预训练一样，所有参数参与微调，区别在于数据是有指令跟随的 QA 数据。
- prompt tuning
- p tuning
- prefix tuning
- Adapter：在神经网络基础上添加残差模块，一般是线性变化或简单神经网络，参数量有限，训练效率高，不太会出现灾难性遗忘问题。在 RAG 实战任务中，可在 q 前面增加 Adapter 提高查询准确率。
- LoRA：矩阵分解，将比如4096*4096的大矩阵（参数量160w）分解为比如4096 * 16和16 * 4096的2个矩阵（参数量13W），下降90%

## 微调效果

- prompt的质量和多样性远重要于数据量级。



## 文献资料

LoRA：

[LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685)

数据准备：

[LIMA: Less Is More for Alignment](https://arxiv.org/pdf/2305.11206)

[Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone](https://arxiv.org/pdf/2404.14219)



## 显存计算和量化



模型应该用多少资源训练？

以BP16算，一个BP16占2个字节

batch_size * seq_len * hidden_dim * 2 / 1024 / 1024

以Llama 13B为例

输入输出：1 * 1024 * 5120 * 2  / 1024 / 1024 ≈ 20MB

参数：26GB （13 * 2）

优化器：13 * 4 （FP32） * 3 （梯度指数平滑值，梯度平方指数平滑值，模型参数） = 156GB

激活值：梯度下降时避免重复计算，需要将一些值存入缓存。与seq_len的平方和batch_size成正比。Llama 13B这部分约为14.5GB

梯度值：13B * 2(FP16) = 26G

共计222.5GB （4*H100）

全量训练所需显存大概是参数量*20 GB

LoRA微调所需显存大概是参数量*2 GB
