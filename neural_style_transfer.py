
# importing libraries
import numpy as np
import cv2
import time
import tensorflow as tf
import keras
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt

# build pretrained model
model = VGG19(include_top = False, weights = 'imagenet')
model.trainable = False
model.summary()

# desired height and weidth
img_height = 400
img_weidth = 600


# load and process image file
def load_and_process_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_weidth))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis = 0)
    return img

# deprocess image file i.e opposite of preprocess image
def deprocess(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint')
    return x

# display image
def display_image(image):
    if len(image.shape) ==4:
        img = np.squeeze(image, axis = 0)
    img = deprocess(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)

# display content and style image
display_image(load_and_process_image('content.jpg'))
display_image(load_and_process_image('style.jpg'))


# layer used for content loss
content_layer = 'block5_conv4'
# layer used for style loss
style_layer = ['block1_conv1','block2_conv1',  'block3_conv1',"block4_conv1",  'block5_conv1']

# building content and style model
content_model = Model(inputs = model.input, outputs = model.get_layer(content_layer).output)
style_models = [Model(inputs = model.input, outputs = model.get_layer(i).output) for i in style_layer]


# compute content cost
def content_cost(content, generated):
    input_c = tf.concat([content, generated],axis = 0)
    features = content_model(input_c)
    cost = tf.reduce_mean(tf.square(features[0, :, :, :]-features[1, :, :, :]))
    return cost

# compute gram matrix
def gram_matrix(x):
    n_C = int(x.shape[-1])
    a = tf.reshape(x, shape = (-1, n_C))
    n = tf.shape(a)[0]
    G = tf.matmul(a, a, transpose_a = True)
    return G/tf.cast(n, tf.float32)


lam = 1./len(style_models)

# compute style cost
def style_cost(style, generated):
    J_style = 0
    input_s = tf.concat([style, generated], axis = 0)
    for style_model in style_models:
        features = style_model(input_s)
        GS = gram_matrix(features[0, :, :, :])
        GG = gram_matrix(features[1, :, :, :])
        current_cost = tf.reduce_mean(tf.square(GS-GG))
        J_style += current_cost*lam
    return J_style


# compute variation cost
def total_variation_loss(x):
    a = tf.square(x[:, :img_height-1, :img_weidth-1, :]-x[:, 1:, 1:, :])
    b = tf.square(x[:, :img_height-1, :img_weidth-1, :]-x[:, :img_height-1, 1:, :])
    return tf.reduce_sum(tf.pow(a+b, 1.25))


# train the model
def training_loop(content_path, style_path, iterations = 20, alpha = 20., beta = 40., gamma = 30):
    content = load_and_process_image(content_path)
    style = load_and_process_image(style_path)
    generated = tf.Variable(content, dtype = tf.float32)
    opt = tf.optimizers.Adam(learning_rate = 15)
    best_cost = 1e12+0.1
    best_image = None
    start_time = time.time()
    generated_images = []
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J_content = content_cost(content, generated)
            J_style = style_cost(style, generated)
            J_var = total_variation_loss(generated)
            J_total = alpha*J_content+beta*J_style + gamma*J_var
        grads = tape.gradient(J_total, generated)
        opt.apply_gradients([(grads, generated)])
        if J_total<best_cost:
            best_cost = J_total
            best_image = generated.numpy()
        print(f'cost at {i} :{J_total} time:{time.time()-start_time}')
        generated_images.append(generated.numpy())
    return best_image, generated_images

# calling our model
best_image, generated_images = training_loop('content.jpg', 'style.jpg')

# display image
display_image(best_image)


# saving the best image generated
keras.preprocessing.image.save_img( f"best_generated_image time:{time.time()}.jpg", cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))






