# NeuralGlobalLighting
用Unet(卷积神经网络)实现间接光照预测，基于mitsuba3，pytorch

# 基本原理
利用Unet预测间接光照图，输入为Gbuffer数据(位置图，法线图，albedo图，直接光照图)，输出为间接光照预测结果

全局光照效果 = 直接光照(用路径追踪生成的，2048SPP，这一步比较慢，可以改用其他更快的方法) + 预测的间接光照*albedo

离线训练Unet，经过测试，模型有一定的泛化能力，用多个场景的训练集训练Unet，可以迁移至新的场景。
# 效果
测试场景 - living-room

![image](https://github.com/user-attachments/assets/b4ee5a64-8022-4848-bd21-3db94f77adb7)

| 原图-4096SPP | 预测间接光照效果 | 伪影 |
| --- | --- | --- |
| ![原图](https://github.com/user-attachments/assets/f35c2e83-7f72-4dc1-8e9e-2841efdbe78c) | ![迭代后](https://github.com/user-attachments/assets/9863c978-5c9a-44bd-91fe-75ad73a4baa4) | ![伪影](https://github.com/user-attachments/assets/8d9f728f-a22d-42ec-b5d5-87221f81b871) |

**图注：**
- **左图**：4096 SPP 的渲染结果。
- **中图**：直接光照+Gbuffer预测的间接光照渲染结果。
- **右图**：神经网络训练过程中，预测的间接光照出现伪影

## 更多效果比较
下面的结果，仅仅比较训练不同轮次的结果，左边是最终效果，中间是神经网络预测的间接光照，右边是实际间接光照值

1. 训练 1 epoch：

![image](https://github.com/user-attachments/assets/25307a6b-579e-44e5-bfdb-5d020fd99b78)

2. 训练 10 epoch：

![image](https://github.com/user-attachments/assets/56ad3357-c4da-45ed-8ee0-7f4419810276)

3. 训练 300 epoch：

![image](https://github.com/user-attachments/assets/c54679d8-1e9f-4dd9-b641-162fde93dffb)

从结果可以看到，随着训练的深入，Unet能够预测出难以分辨的间接光照效果！

再看看再其他场景下的预测效果（主要是看看cornel box下的color bleeding能不能预测）：
1. 训练 1 epoch：

![image](https://github.com/user-attachments/assets/aeb6cf84-395e-410a-b68d-cc4dd549a4dc)

2. 训练 10 epoch：

![image](https://github.com/user-attachments/assets/1663060c-f2e4-42e3-a10a-680721ede701)

可以发现，这种方法能够预测 color bleeding 这种比较复杂的效果！这要归功于Unet，卷积神经网络可以扩大一个像素的感受野！

## 多场景测试
这里为了的图，直接各个场景处理后的最终结果。Unet的训练集从单场景扩展到多场景：
1. 训练 1 epoch：

![image](https://github.com/user-attachments/assets/a0109068-ea42-4e08-88a4-2c3ff15e7f6a)

2. 训练 10 epoch：

![image](https://github.com/user-attachments/assets/584cc1f7-8eee-4625-97aa-1ee775fd12b6)

3. 训练 188 epoch:

![image](https://github.com/user-attachments/assets/feba1255-10fe-4381-abbc-a5ae36b2da85)

从结果来看，泛化能力不是很强，当然也可能是数据集比较小，还可以进一步提高。




# 优缺点分析
优点：实现简单，一个Unet就可以完成所有像素间接光照值的预测
缺点：容易出现一些伪影，在一些阴影上效果差一些
