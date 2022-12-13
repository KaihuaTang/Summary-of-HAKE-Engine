# Summary-of-HAKE-Engine
HAKE引擎相关论文代码整理

## 目录
1. [HAKE引擎背景](#hake引擎背景)
2. [PaStaNet(HAKE)模型部署](#pastanethake模型部署)


## HAKE引擎背景
[HAKE引擎](http://hake-mvig.cn/home/)是由上海交通大学[卢策吾老师](https://www.mvig.org/)团队的[李永露老师](https://dirtyharrylyl.github.io/)主导开发的行为识别数据集与代码框架。其主要目的是服务于人的行为识别，具体为人与物体的交互识别[(Human-Object Interaction / HOI)](https://github.com/DirtyHarryLYL/HOI-Learning-List)。

根据其项目主页，该项目包含了多篇论文与工作。经总结，可归纳为：(1) [PaStaNet/HAKE (CVPR2020)](https://github.com/DirtyHarryLYL/HAKE), (2) [TIN: Transferable Interactiveness Network (CVPR2019,TPAMI2022)](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network), (3) [2D-3D Matching (CVPR2020)](https://github.com/DirtyHarryLYL/DJ-RN), (4) [Attribute-Object Composition (CVPR2020)](https://github.com/DirtyHarryLYL/SymNet), (5) [HOI Analysis (NeurIPS2020)](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)), (6) [Interactiveness Field (CVPR2022)](https://github.com/Foruck/Interactiveness-Field), (7) [PartMap (ECCV2022)](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness).

本文档主要用于指导上述论文中所提供模型的部署和测试。

## PaStaNet(HAKE)模型部署
论文链接：[PaStaNet](https://arxiv.org/abs/2004.00945), [HAKE2.0(期刊拓展)](https://arxiv.org/abs/2202.06851)

代码链接：[HAKE-Action(pytorch)](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec), [HAKE-Action(tensorflow)](https://github.com/DirtyHarryLYL/HAKE-Action)

数据集链接：[HAKE数据集](https://github.com/DirtyHarryLYL/HAKE), [HAKE2.0数据集拓展(额外AVA数据)](https://github.com/DirtyHarryLYL/HAKE-AVA)




