import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# Load images
content_path = "content.jpg"
style_path = "style.jpg"

# Function to load image
def load_img(path):
    img = PIL.Image.open(path)
    img = img.resize((512, 512))
    img = np.array(img)

    img = img.astype(np.float32)
    img = img[np.newaxis, :]

    return tf.convert_to_tensor(img)

content_image = load_img(content_path)
style_image = load_img(style_path)

# Load VGG19 model
model = tf.keras.applications.VGG19(
    include_top=False,
    weights='imagenet'
)

model.trainable = False

# Layers
content_layers = ['block5_conv2']

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

# Create feature extractor
outputs = [model.get_layer(name).output
           for name in (style_layers + content_layers)]

feature_extractor = tf.keras.Model(
    [model.input],
    outputs
)

# Preprocess
def preprocess(image):
    image = tf.keras.applications.vgg19.preprocess_input(image)
    return image

# Extract features
content_features = feature_extractor(
    preprocess(content_image)
)

style_features = feature_extractor(
    preprocess(style_image)
)

# Show images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Content Image")
plt.imshow(np.squeeze(content_image).astype('uint8'))

plt.subplot(1,2,2)
plt.title("Style Image")
plt.imshow(np.squeeze(style_image).astype('uint8'))

plt.show()

print("Neural Style Transfer Setup Complete!")