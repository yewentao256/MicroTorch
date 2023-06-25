# TODO

大项：

- 单元测试
- grad删除，改成正式的require_grad相关逻辑
- 目录结构重构，h和cpp、cuda文件拆分
- 支持cuda算子：sub、mul等剩余cuda算子
- 可选：引入注册op，执行op机制（带队列——引擎）
- tensor功能丰富：包括view、contiguous、索引、切片等操作
- 将tensor升级为二维支持(shape、取值等)
    已知难以将取值取数组pybind导出重载，这就需要我们实现python层处理参数了

小项：

- 调研并理解pytorch公式 [https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD]
- `valgrind --leak-check=full`检查内存泄漏？
