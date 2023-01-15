# TODO

大项：

- 支持cuda算子
    支持一个最简单的add算子，完整编译（提示：传到cuda的是data指针，不需要穿tensor类）
    支持编译时配置是否使用cuda算子
    支持剩余所有算子
    拆分cuda文件和h文件，让目录结构更好
- 引入注册op，执行op机制（带队列——引擎）
- cuda算子再引入一个队列机制

小项：

- 调研并理解pytorch公式 https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD
- Tensor支持python进行切片操作
- 画一个图表示调用链路
- 锁定pybind11版本（不需要submodule）