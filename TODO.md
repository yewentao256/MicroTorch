# TODO

大项：

- 支持cuda算子
    当前add BACKWARD cuda有bug
    所有out应该是infer好传参进去，而不是return返回——注意不能在临时局部变量里这么操作，应该有一个中间类存储所有数据（也便于之后开发引擎）
    拆分cuda文件和h文件，让目录结构更好
- 引入注册op，执行op机制（带队列——引擎）
- cuda算子再引入一个队列机制
- 重构tensor，不再使用std::vector
    按照https://zhuanlan.zhihu.com/p/340228853 和simple tensor逐步支持
    包括transpose、view、contiguous、索引、切片等操作
    记得加上单元测试（如测试取值等)
    最后把std vector换成float*
- 将tensor升级为二维支持(shape、取值等)
    已知难以将取值取数组pybind导出重载，这就需要我们实现python层处理参数了

小项：

- 调研并理解pytorch公式 [https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD]
- Tensor支持python进行切片操作
- `valgrind --leak-check=full`检查内存泄漏？