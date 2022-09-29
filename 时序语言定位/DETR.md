# Highlight

将目标检测视作一个端到端的集合预测问题，利用Transfomer的自注意力机制，去除了传统目标检测的pipeline中的生成锚框和非极大值抑制（NMS）等人工模块

# Architecture

 ![](E:\Github\Notes\_resources\2022-09-29-09-48-26-image.png)

## Backbone

输入图像$\mathcal{R}^{3\times H_0\times W_0}$，使用CNN模块下采样为特征图$\mathcal{R}^{C\times H \times W}$，其中$H,W=\frac{H_0}{32}, \frac{W_0}{32}$

## Transformer Encoder

使用$1×1$卷积将特征图降为$d$维，然后展平为$d \times HW$维的token矩阵，加上位置编码后，输入到transformer编码器中，得到$d \times HW$维的编码表征$z$

## Transformer Decoder

使用N个query向量作为查询向量，得到N个输出向量，表示N个候选框（N要远大于一张图中的目标个数）

* Tricks
  
  * query生成Q，$z$生成K和V
  
  * query本身充当每一层的Position Embedding

## Prediciton Feed-forward Networks

两个相同的FFN，一个用于预测类别，一个用于预测坐标（归一化的中心坐标和长宽）

<img title="" src="file:///E:/Github/Notes/_resources/2022-09-29-14-30-05-image.png" alt="" width="430" data-align="center">

# Trainng Loss

现有大小为N的候选标签集合 $\hat{y}$ 和大小为M的标签集合$y$（N>M），**关键问题是：如何给N个候选框分配标签**

1. 给标签集合 $y$ 填充背景框（类别为Background，坐标全为0）至大小为N

2. 通过改变$\hat{y}$的排列顺序，找到$y$和$\hat{y}$之间的最优配对
   
   假设$\sigma$表示一种顺序，使用匈牙利算法，求解出最优排序

$$
\hat{\sigma}=\underset{\sigma \in \mathfrak{S}_N}{\arg \min } \sum_i^N \mathcal{L}_{\operatorname{match}}\left(y_i, \hat{y}_{\sigma(i)}\right)
$$

       其中$\mathcal{L}_{\operatorname{match}}\left(y_i, \hat{y}_{\sigma(i)}\right)=-\mathbb{1}_{\left\{c_i \neq \varnothing\right\}} \hat{p}_{\sigma(i)}\left(c_i\right)+\mathbb{1}_{\left\{c_i \neq \varnothing\right\}} \mathcal{L}_{\mathrm{box}}\left(b_i, \hat{b}_{\sigma(i)}\right)$ ，即考虑所有的非背景框，预测错误概率和与标签坐标之间的偏差

3. 对$\hat{y}$进行最优排序后，计算训练的损失函数

$$
\mathcal{L}_{\text {Hungarian }}(y, \hat{y})=\sum_{i=1}^N\left[-\log \hat{p}_{\hat{\sigma}(i)}\left(c_i\right)+\mathbb{1}_{\left\{c_i \neq \varnothing\right\}} \mathcal{L}_{\mathrm{box}}\left(b_i, \hat{b}_{\hat{\sigma}}(i)\right)\right]
$$

**匹配时，类别判定只考虑正类，意味着只要ground truth能对应好就行，padding框怎么排序无所谓。训练时，类别判定要考虑正负类，因为希望模型能正确地将padding框预测为背景**

4. Bounding box loss

$$
\mathcal{L}_{\mathrm{box}}\left(b_i, \hat{b}_{\hat{\sigma}}(i)\right)=\lambda_{\mathrm{iou}} \mathcal{L}_{\mathrm{iou}}\left(b_i, \hat{b}_{\sigma(i)}\right)+\lambda_{\mathrm{L} 1}\left\|b_i-\hat{b}_{\sigma(i)}\right\|_1
$$

其中$\mathcal{L}_{\mathrm{iou}}$为GIoU损失

L1 Loss的数值与目标尺寸有关，不能用来衡量目标之间的相对偏差，因此不能仅用L1 Loss来作为Bounding box loss
