# Highlight

多模态理解系统需要使用预训练的目标检测器来对应文本概念和视觉目标，但先前的方法都会在训练时将目标检测器固定，这种做法会导致两个问题：

1. 目标检测器的类别空间限制了整个推理系统的上限

2. 使用固定的目标检测器得到检测区域后，再开始推理，限制了整个系统的视野

本文将DETR作为可训练的目标检测模块，提出了一个真正端到端的多模态推理系统，使用名词短语和检测框的对齐作为预训练的监督信号。

# Model

![](E:\Github\Notes\_resources\2022-09-30-11-11-01-image.png)

视觉和文本分别编码后，在序列维度上进行拼接（即同时作为tokens）输入到DETR，最终解码得到N个预测框

## Pretraining

### Dataset

构造了一个图文对预训练数据集，标注好了目标框和句中token的对应关系

### Soft token prediction

![](E:\Github\Notes\_resources\2022-09-30-12-45-45-image.png)

匈牙利匹配只能是一对一的，而目标和词组是多对多关系。所以此处是直接对序列中的tokens进行多标签预测

设定每个句子最大长度为256（在句尾填充$\phi$字符表示无目标），然后将目标token的标签设置为1，其余设置为0，使用交叉熵进行训练

### Contrastive alignment

直接用配对的目标和token来进行对比学习，使用InfoNCE Loss

假设某个目标$o_i$对应的token集合为$T_i^+$

$$
l_o=\sum_{i=0}^{N-1} \frac{1}{\left|T_i^{+}\right|} \sum_{j \in T_i^{+}}-\log \left(\frac{\exp \left(o_i^{\top} t_j / \tau\right)}{\sum_{k=0}^{L-1} \exp \left(o_i^{\top} t_k / \tau\right)}\right)
$$

假设某个token $t_i$对应的目标集合为

$$
l_t=\sum_{i=0}^{L-1} \frac{1}{\left|O_i^{+}\right|} \sum_{j \in O_i^{+}}-\log \left(\frac{\exp \left(t_i^{\top} o_j / \tau\right)}{\sum_{k=0}^{N-1} \exp \left(t_i^{\top} o_k / \tau\right)}\right)
$$

## Downstream Task

### Phrase grounding

- 给定图像和词组，输出每个词组所对应的目标框

- 直接使用Soft token prediction来微调即可

### Referring expression comprehension

- 给定图像和一段文本，输出整段文本所描述的目标框

- 微调时，只根据$\phi$的预测概率进行排序，$1-P(\phi)$即为预测概率

### Referring expression segmentation

- 给定图像和一段文本，输出整段文本所描述的目标的分割图像

- 两阶段微调
  
  1. 训练预测框
  
  2. 固定模型，训练分割头，对预测框的图像进行分割

### Visual Question Answering

![](E:\Github\Notes\_resources\2022-09-30-13-30-57-image.png)

- 给定图像和一条问题，输出答案，和答案依据的目标框

- 微调时，使用一部分query用于预测框坐标，另一部分用于回答问题
