import os
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, LeakyReLU, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Model parametreleri
img_rows = 256
img_cols = 256
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

# Verilerin yüklenmesi
def load_images_from_folder(folder, img_rows, img_cols):
    images = []
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Klasör bulunamadı: {folder}")
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, color_mode='grayscale', target_size=(img_rows, img_cols))
        if img is not None:
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

# Mevcut görüntülerin bulunduğu klasörün yolu
dex_image_folder = r"D:\\Mezuniyet Projesi\\datasets\\samples\\benign\\beningmanifestpng"
X_train = load_images_from_folder(dex_image_folder, img_rows, img_cols)

# Normalizasyon
X_train = X_train / 127.5 - 1.0

# DCGAN modeli
def build_generator():
    model = Sequential()
    model.add(Dense(256 * 64 * 64, activation="relu", input_dim=latent_dim))
    model.add(Reshape((64, 64, 256)))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(channels, kernel_size=4, strides=1, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
    discriminator.trainable = False

    noise = Input(shape=(latent_dim,))
    img = generator(noise)
    validity = discriminator(img)

    combined = Model(noise, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    return combined

# Modelin oluşturulması
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)

# Modelin eğitilmesi
def train(epochs, batch_size=8, save_interval=20, num_images_to_save=64):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch(noise, valid)

        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_imgs(epoch, generator, num_images_to_save)
            print(f"Epoch {epoch}: Görüntüler kaydedildi.")

def save_imgs(epoch, generator, num_images_to_save):
    save_dir = r"D:\\Mezuniyet Projesi\\datasets\\samples\\benign\\dcgan\\dcgandenememanifest"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    noise = np.random.normal(0, 1, (num_images_to_save, latent_dim))
    gen_imgs = generator.predict(noise)

    for i in range(gen_imgs.shape[0]):
        img = gen_imgs[i]
        img = 0.5 * img + 0.5  # Normalize the image to [0, 1]
        img_path = os.path.join(save_dir, f"epoch{epoch}_img{i}.png")
        plt.imsave(img_path, img[:, :, 0], cmap='gray')

# Eğitim
train(epochs=1000, batch_size=8, save_interval=20, num_images_to_save=64)
