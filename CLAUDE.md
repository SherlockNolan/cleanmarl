## 项目概述

本项目是对水下潜器多智能体强化学习进行研究。使用的Unity ML-Agents接口。主要是从Unity搭建强化学习环境，然后导出binary build文件接入别人开源实现的python的[cleanmarl](https://github.com/AmineAndam04/cleanmarl)算法进行多智能体强化学习。

目前实现的unity多智能体强化学习的环境有很多。预计需要研究：
- **3chase1 (herding)** 三追一问题，定义为一个Herder，两个Chaser/Netter去追一个逃方Prey。Herder如同渔民一个人赶网，两个Chaser手拉手拉网，将鱼进行捕获。
- **3chase1 (chasing)** 三追一，但是是三个人手拉手拉网，在一个立体水下空间对Prey进行追赶。
- **2chase1** 二追一问题，两个chaser拉网追一个Prey

目前初步的设想是让Herder/Chaser/Netter的最大速度比Prey慢，但是瞬时加速度比Prey快，能利用合作关系巧妙追上。

### 文件夹结构

cleanmarl/
  --archive/ 原来的一些其它算法的实现被我归入这个文件夹整理
  --env/ 共用的env环境

checkpoints/ 保存RL模型

docs/ 文档和使用说明

目前正在做第一个3chase1 (herding), 写了`cleanmarl/mappo_3chase1_unity.py`这个文件。

## 个性化

使用英文思考，回答推荐使用中文

对于新写的单个CLI脚本，在脚本顶端写上文件注释和用法说明。没有明确要求的时候不添加md文档。

遵循良好的设计模式规范