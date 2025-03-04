# NeuralGlobalLighting
用Unet(卷积神经网络)实现间接光照预测，基于mitsuba3，pytorch

# 基本原理
利用Unet预测间接光照图，输入为Gbuffer数据(位置图，法线图，albedo图，直接光照图)，输出为间接光照预测结果

全局光照效果 = 直接光照(用路径追踪生成的，2048SPP，这一步比较慢，可以改用其他更快的方法) + 预测的间接光照*albedo

离线训练Unet，经过测试，模型有一定的泛化能力，用多个场景的训练集训练Unet，可以迁移至新的场景。
# 效果
测试场景 - living-room

![image](https://github.com/user-attachments/assets/b4ee5a64-8022-4848-bd21-3db94f77adb7)


Unet训练200多轮后，在living-room场景下达到的预测效果：

![image](https://github.com/user-attachments/assets/9863c978-5c9a-44bd-91fe-75ad73a4baa4)

有时候会出现一些伪影：

![image](https://github.com/user-attachments/assets/8d9f728f-a22d-42ec-b5d5-87221f81b871)

# 优缺点分析
优点：实现简单，一个Unet就可以完成所有像素间接光照值的预测
缺点：容易出现一些伪影，在一些阴影上效果差一些
