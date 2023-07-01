# TODO

大项：

- 单元测试
- 目录结构重构，h和cpp、cuda文件拆分
- 支持cuda算子：sub、mul等剩余cuda算子
- tensor功能丰富：包括view、contiguous、索引、切片等操作
- 将tensor升级为二维支持(shape、取值等)
    已知难以将取值取数组pybind导出重载，这就需要我们实现python层处理参数了
- lint（包括python和cpp）
- 支持广播：注意不仅仅是前向计算，求导时，比如z = x(3,3) + y(3,1)；y被广播后每个 y 的元素都参与了三次相加操作。所以在反向传播时，这三次操作的梯度（比如每次都是 1）会被累加。
- 可选：tensor 支持dtype
- 可选（可能不开发）：引擎执行异步逻辑

小项：

- 调研并理解pytorch公式 [https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html#SGD]
- `valgrind --leak-check=full`检查内存泄漏？
- 声明的tensor应该是未初始化的。
