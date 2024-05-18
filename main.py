import tensorflow as tf
from keras.src.datasets import fashion_mnist
from matplotlib import pyplot as plt
from tensorflow.keras import layers, models, optimizers

# 加载数据集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)) / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)) / 255.0
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


# 构建VGGNet模型
def build_vggnet_model(learning_rate):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# 构建ResNet模型
def build_resnet_model(learning_rate):
    input_layer = layers.Input(shape=(28, 28, 1))

    # Initial Convolutional Layer
    x = layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu')(input_layer)

    # Residual Blocks
    x = residual_block(x, 16)
    x = residual_block(x, 32)
    x = residual_block(x, 64)
    x = residual_block(x, 128)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Output Layer
    output_layer = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def residual_block(input_layer, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Adjusting the shape of input_layer to match the shape of x
    if input_layer.shape[-1] != filters:
        input_layer = layers.Conv2D(filters, (1, 1), padding='same')(input_layer)

    x = layers.Add()([input_layer, x])
    x = layers.Activation('relu')(x)
    return x


def plot(history):
    # 绘制损失曲线
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 绘制准确率曲线
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# 设置参数范围
learning_rates = [0.01, 0.001]
network_structures = ['vggnet', 'resnet']

# 循环遍历不同的学习率和网络结构进行训练和测试
for lr in learning_rates:
    for structure in network_structures:
        if structure == 'vggnet':
            # 构建并编译VGGNet模型
            model = build_vggnet_model(learning_rate=lr)
        elif structure == 'resnet':
            # 构建并编译ResNet模型
            model = build_resnet_model(learning_rate=lr)

        # 训练模型
        history = model.fit(train_images, train_labels, epochs=100, batch_size=64,
                            validation_data=(test_images, test_labels), verbose=1)
        # 测试模型
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

        # 打印测试结果
        print(f"Learning Rate: {lr}")
        print(f"Network Structure: {structure}")
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_acc}")

        # 绘制曲线
        plot(history)
