# TODO

大项：

- cuda runtime.h使用要考虑一下如何接入。现在解了std vector的依赖，所以初始化上可能要python层做一些事情，然后把单元测试跑起来
- grad删除，改成正式的require_grad相关逻辑
- 支持cuda算子
    拆分cuda文件和h文件，让目录结构更好
- 引入注册op，执行op机制（带队列——引擎）
- 重构tensor，不再使用std::vector
    包括transpose、view、contiguous、索引、切片等操作
    记得加上单元测试（如测试取值等
    tinytorch .cuda()的时候就应该给这个tensor分配内存！
- 将tensor升级为二维支持(shape、取值等)
    已知难以将取值取数组pybind导出重载，这就需要我们实现python层处理参数了

小项：

- 调研并理解pytorch公式 [https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD]
- Tensor支持python进行切片操作
- `valgrind --leak-check=full`检查内存泄漏？