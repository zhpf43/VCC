import numpy as np
import tensorflow as tf
import time
import keras
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, r=16):
        super(SEBlock, self).__init__()
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reshape = tf.keras.layers.Reshape((1, 1, input_channels))
        self.fc1 = tf.keras.layers.Dense(units=input_channels // r)
        self.fc2 = tf.keras.layers.Dense(units=input_channels)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.reshape(x)
        x = self.fc1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.sigmoid(x)
        return inputs * x


class ASPP(tf.keras.layers.Layer):
    def __init__(self, num_classes, rates=[6, 12, 18, 24], **kwargs):
        super(ASPP, self).__init__(**kwargs)
        self.aspp_blocks = []
        for rate in rates:
            conv = tf.keras.layers.Conv2D(
                256, (3, 3), dilation_rate=rate, padding="same"
            )
            bn = tf.keras.layers.BatchNormalization()
            relu = tf.keras.layers.ReLU()
            self.aspp_blocks.append([conv, bn, relu])
        self.gap = tf.keras.layers.GlobalAveragePooling2D()  # 全局平均池化
        self.reshape = tf.keras.layers.Reshape((1, 1, -1))
        self.upsample = tf.keras.layers.UpSampling2D(
            (32, 32)
        )  # 将全局平均池化的结果上采样到(32, 32)
        self.final_conv = tf.keras.layers.Conv2D(
            num_classes, (1, 1), padding="same"
        )  # 添加最后的卷积层

    def call(self, inputs):
        res = []
        for block in self.aspp_blocks:
            conv, bn, relu = block  # 依次经过卷积、批量归一化、激活函数
            x = conv(inputs)
            x = bn(x)
            x = relu(x)
            x = tf.image.resize(x, (32, 32))  # 将ASPP模块的结果上采样到(32, 32)
            res.append(x)
        gap = self.gap(inputs)
        gap = self.reshape(gap)
        gap = self.upsample(gap)
        res.append(gap)
        x = tf.concat(res, axis=-1)
        x = self.final_conv(x)  # 将结果传递给最后的卷积层
        return x


def vgg_block(num_convs, num_channels):  # 自定义卷积层数目，卷积核数目
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(
                num_channels,
                kernel_size=3,
                padding="same",
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            )
        )
        blk.add(tf.keras.layers.BatchNormalization())  # 添加批量归一化层
    blk.add(SEBlock(num_channels))  # 添加SEBlock
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


def build_vggnet(conv_arch):
    net = tf.keras.models.Sequential()
    for num_convs, num_channels in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
        net.add(ASPP(10))
    net.add(
        tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(
                    4096,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    4096,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )
    )
    return net


def build_vgg(keyword="vgg11"):
    if keyword == "vgg11":
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # VGG11
    if keyword == "vgg16":
        conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # VGG16
    if keyword == "vgg19":
        conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))  # VGG19
    net = build_vggnet(conv_arch)
    return net


# 数据加载器
class DataLoader:
    def __init__(self):
        initial_data = tf.keras.datasets.fashion_mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = (
            initial_data.load_data()
        )
        # 将图片转为float32且除255进行归一化；expand_dims增加维度
        self.train_images = np.expand_dims(
            self.train_images.astype(np.float32) / 255.0, axis=-1
        )
        self.test_images = np.expand_dims(
            self.test_images.astype(np.float32) / 255.0, axis=-1
        )
        self.train_labels = self.train_labels.astype(np.int32)
        self.test_labels = self.test_labels.astype(np.int32)
        self.num_train, self.num_test = (
            self.train_images.shape[0],
            self.test_images.shape[0],
        )

    # 获取训练集和测试集的batch
    def get_batch_train(self, batch_size):
        # 从训练集中随机产生batch_size个索引
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        # 将图片resize至合适大小
        resized_images = tf.image.resize_with_pad(
            self.train_images[index],
            32,
            32,
        )
        # 数据增强
        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.RandomFlip(
                    "horizontal_and_vertical"
                ),  # 随机水平翻转和垂直翻转
                tf.keras.layers.experimental.preprocessing.RandomRotation(
                    0.2
                ),  # 随机旋转
            ]
        )
        resized_images = data_augmentation(resized_images)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        resized_images = tf.image.resize_with_pad(
            self.test_images[index],
            32,
            32,
        )
        return resized_images.numpy(), self.test_labels[index]


# 循环学习率
def cyclic_lr(epoch):
    base_lr = 0.000001  # 基础学习率
    max_lr = 0.0001  # 最大学习率
    step_size = 2000.0  # 两次最大学习率之间的步数
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(cyclic_lr)


# 主控程序，调用数据并训练模型
# 定义超参数
num_epochs = 100  # 训练次数
batch_size = 80  # 每次训练的图片数量
learning_rate = 0.000001


# 初始化最佳验证损失和早停计数器
best_val_loss = float("inf")
early_stopping_counter = 0
patience = 5  # 早停轮数4

# 绘图
summary_writer = tf.summary.create_file_writer(
    "C:/Users/lenovo/Desktop/summary_writerssss"
)
print(
    "now begin the train, time is {}".format(
        time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    )
)
model = build_vgg("vgg11")

data_loader = DataLoader()
optimier = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train // batch_size * num_epochs)
num_per_epoch = 0
num_per_epochs = 0

for batch_index in range(num_batches):
    X, y = data_loader.get_batch_train(batch_size)  # X输入数据，y是对应标签
    with tf.GradientTape() as tape:
        y_pred = model(X)  # 获取预测值
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_sum(loss)
    # print("batch %d: loss %f" % (batch_index, loss.numpy()))

    # 记录训练损失
    with summary_writer.as_default():
        tf.summary.scalar("train_loss", loss, step=batch_index)

    # 计算损失相对于模型参数的梯度
    grads = tape.gradient(loss, model.variables)
    optimier.apply_gradients(
        grads_and_vars=zip(grads, model.variables)
    )  # 将计算出的梯度用于模型参数上
    num_per_epoch += 1

    #  早停
    if num_per_epoch >= int(data_loader.num_train // batch_size):
        num_per_epochs += 1
        print("epoch:", num_per_epochs)
        print("loss:", loss.numpy())
        print("|||||||||||||||||||||||||||||||||||")
        num_per_epoch = 0

        # 在每个epoch结束时计算测试集的损失
        val_loss = 0
        val_batches = 0
        for batch_index in range(int(data_loader.test_images.shape[0] // batch_size)):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            val_X, val_y = data_loader.get_batch_test(batch_size)
            y_pred = model(val_X)
            val_loss += tf.reduce_sum(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=val_y, y_pred=y_pred
                )
            ).numpy()
            val_batches += 1
        val_loss /= val_batches

        # 记录学习率
        with summary_writer.as_default():
            tf.summary.scalar("test_loss", val_loss, step=num_per_epochs)

        # 如果验证损失没有改善，增加早停计数器
        if val_loss >= best_val_loss:
            early_stopping_counter += 1
        else:
            best_val_loss = val_loss
            early_stopping_counter = 0

        # 如果早停计数器达到patience，停止训练
        if early_stopping_counter >= patience:
            break

        # 每个epoch结束时重设优化器学习率
        new_lr = cyclic_lr(num_per_epochs)  # 计算新学习率
        optimier.learning_rate.assign(new_lr)  # 更新优化器学习率
        print("Now learing rate: ", new_lr)

print("now end the train, time is ")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# 模型的评估
# 创建一个空列表来保存所有的预测结果和真实标签
all_predictions = []
all_labels = []
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches_test = int(
    data_loader.num_test // 30
)  # 把测试数据拆分成多批次，每个批次30张图片
print("now begin the test, time is ")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for batch_index in range(num_batches_test):
    test_images, test_labels = data_loader.get_batch_test(30)
    y_p = model.predict(test_images)
    sparse_categorical_accuracy.update_state(y_true=test_labels, y_pred=y_p)
    all_predictions.extend(np.argmax(y_p, axis=1))
    all_labels.extend(test_labels)

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_predictions)

# 绘制混淆矩阵
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("True")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()

print("test accuracy: %f" % sparse_categorical_accuracy.result())
print("now end the test, time is ")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
