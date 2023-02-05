# TODO

大项：

- 支持cuda算子
    当前add BACKWARD cuda有bug
    ~~支持编译时配置是否使用cuda算子~~  需求不对，应该是支持运行时是否指派cuda算子！
    当前有一个bug：如果cuda不为on会找不到tinytorch::add_cuda_impl(tinytorch::Tensor, tinytorch::Tensor)（预计dispatch 宏完成后能修复）
    支持剩余所有算子
    拆分cuda文件和h文件，让目录结构更好
- 引入注册op，执行op机制（带队列——引擎）
- cuda算子再引入一个队列机制
- 重构tensor，不再使用std::vector
    按照https://zhuanlan.zhihu.com/p/340228853和simple tensor逐步支持
    包括transpose、view、contiguous、索引、切片等操作
    记得加上单元测试（如测试取值等)
    最后把std vector换成float*

小项：

- 调研并理解pytorch公式 [https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD]
- Tensor支持python进行切片操作
- `valgrind --leak-check=full`检查内存泄漏？
- 基于win10版本测试一下gdb是否可用，1984上不可用看起来是因为临时目录