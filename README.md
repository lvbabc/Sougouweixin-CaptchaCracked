# Sougouweixin-CaptchaCracked
本项目通过 Keras 搭建一个深度卷积神经网络来识别搜狗微信验证码，建议使用显卡来运行该项目。
代码已写成 python 脚本。采用了两种模型来破解验证码，cnn 准确率为90%，ctc准确率为95%（测试万次结果）。

# cnn
##  数据生成器
本项目数据从网上打码平台付费采集，采集 6000 数据共花费800元。数据经筛选后保存在本地，运行时一起读取进程序。
搜狗微信验证码为 6 位数字与大写字母结合。

```
# 0-9 + ABC...Z
characters = string.digits + string.ascii_uppercase
```

## X

X 的形状是 (batch_size, height, width, 3)，一批生成32个样本，图片宽度为140，高度为44，那么形状就是 (32, 44, 140, 3)，取第一张图就是 X[0]。

## y
y 的形状是 6 个 (batch_size, n_class)，如果转换成 numpy 的格式，则是 (n_len, batch_size, n_class)，比如一批生成32个样本，验证码的字符有36种，长度是6位，那么它的形状就是4个 (32, 36)
```
folder = '数据集目录'
assert os.path.exists(folder)
assert os.path.isdir(folder)
imageList = os.listdir(folder)
imageList = [os.path.join(folder, item) for item in imageList if os.path.isfile(os.path.join(folder, item))]
image_size = len(imageList)

image_array = np.zeros((image_size, height, width, 3), dtype=np.uint8)
image_names = []
for i in range(image_size):
    img = Image.open(imageList[i])
    image_array[i] = img
    image_names.append(imageList[i][-11:-5].upper())


def gen(batch_size=32):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    while True:
        for i in range(batch_size):
            num = random.randint(0, image_size-1)
            random_str = image_names[num]
            X[i] = image_array[num]
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y
```

# 微信
lv1559744776
