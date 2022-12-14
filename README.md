# Summary-of-HAKE-Engine-and-Some-Cross-Modality-Papers
HAKE引擎相关论文以及Noah近期多模态论文代码整理

## HAKE引擎背景
[HAKE引擎](http://hake-mvig.cn/home/)是由上海交通大学[卢策吾老师](https://www.mvig.org/)团队的[李永露老师](https://dirtyharrylyl.github.io/)主导开发的行为识别数据集与代码框架。其主要目的是服务于人的行为识别，具体为人与物体的交互识别[(Human-Object Interaction / HOI)](https://github.com/DirtyHarryLYL/HOI-Learning-List)。

根据其项目主页，该项目包含了多篇论文与工作。经总结，可归纳为：(1) [PaStaNet/HAKE (CVPR2020)](https://github.com/DirtyHarryLYL/HAKE), (2) [TIN: Transferable Interactiveness Network (CVPR2019,TPAMI2022)](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network), (3) [2D-3D Matching (CVPR2020)](https://github.com/DirtyHarryLYL/DJ-RN), (4) [Attribute-Object Composition (CVPR2020)](https://github.com/DirtyHarryLYL/SymNet), (5) [HOI Analysis (NeurIPS2020)](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)), (6) [Interactiveness Field (CVPR2022)](https://github.com/Foruck/Interactiveness-Field), (7) [PartMap (ECCV2022)](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness).

本文档主要用于指导上述论文中所提供模型的部署和测试。


## 1. PaStaNet(HAKE)模型部署
- [PaStaNet(HAKE)模型部署指南](HAKE.md)

## 2. TIN模型部署
- [TIN模型部署指南](TIN.md)

## 3. 2D-3D Matching模型部署
该模型主要贡献为Loss，并没有提出新的网络模块。所有模块皆为传统卷积层与线性层。

可参考文件：https://github.com/DirtyHarryLYL/HAKE-Action-Torch/blob/DJ-RN-Torch/hakeaction/models/DJRN.py

## 4. Attribute-Object Composition模型部署
- [Attribute-Object Composition模型部署指南](AOC.md)

因为模块比较简单，该指南直接提取并介绍其设计的模块，而不再赘述环境配置与数据处理。

## 5. HOI Analysis模型部署
- [HOI Analysis模型部署指南](HOIA.md)

该模型主要用AutoEncoder来解决人物交互HOI任务，其创新点也主要侧重于loss的设计。
其核心AutoEncoder模块由Linear层与BatchNorm1d构成，特征转换模块由Transformer构成，并无其他模块。
因此上述指南直接提取并介绍其设计的网络模块，不再赘述环境配置与数据处理。

## 6. Interactiveness Field模型部署
目前该项目仅提供部分代码，且没有提供配置信息，暂时无法跑通。
https://github.com/Foruck/Interactiveness-Field

不过该代码主要基于DETR，可从[文件](https://github.com/Foruck/Interactiveness-Field/blob/main/models/hoi.py)中看出其主要模块除了DETR中的Transformer外主要为MLP与Linear

## 7. PartMap模型部署
- [PartMap模型部署指南](PartMap.md)


==============================  分割线  ============================== 

==============================  分割线  ============================== 

## 诺亚跨模态论文整理

我们同时整理的一些诺亚近期开源的多模态论文，包含：（1）[FILIP: Fine-grained Interactive Language-Image Pre-Training](https://arxiv.org/abs/2111.07783), (2)[Wukong: 100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/Noah_WuKong), (3) [Open-world Semantic Segmentation via Contrasting and Clustering Vision-Language Embedding](https://arxiv.org/abs/2207.08455), (4) [Generative Negative Text Replay for Continual Vision-Language Pretraining](https://arxiv.org/abs/2210.17322), (5) [DetCLIP: Dictionary-Enriched Visual-Concept Paralleled Pre-training for Open-world Detection](https://arxiv.org/abs/2209.09407)

## 1. FILIP: Fine-grained Interactive Language-Image Pre-Training
本文为CLIP模型的改进版本，因为其主要贡献皆为预训练方式，并没有提供新的模型（仅使用已有Transformer模型）故而不作详细展开。

## 2. Wukong大规模中文数据集
- [Wukong大规模中文数据集部署指南](Wukong.md)

虽然本文主要贡献为数据集，但是本文依然提供了一个新的Token Reduction Layer模块，上述链接对该模块会做详细介绍。

## 3. Open-world Semantic Segmentation开放世界分割
暂未找到开源代码，可咨询xuhang

## 4. Generative Negative Text Replay
暂未找到开源代码，可咨询xuhang

## 5. DetCLIP开放世界检测
暂未找到开源代码，可咨询xuhang

