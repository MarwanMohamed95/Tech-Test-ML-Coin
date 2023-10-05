from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import tensorflow as tf


# This function Adds 2 convolutional layers with the parameters passed to it
def conv2d_block(input_tensor, n_filters, kernel_size = 3):

  x = input_tensor
  for i in range(2):
    x = tf.keras.layers.Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
            kernel_initializer = 'he_normal', padding = 'same')(x)
    x = tf.keras.layers.Activation('relu')(x)

  return x

def bottleneck(inputs):

  bottle_neck = conv2d_block(inputs, n_filters=1024)

  return bottle_neck


# defines the one decoder block of the UNet
def decoder_block(inputs, conv_output, n_filters):

  u = tf.keras.layers.Conv2DTranspose(n_filters, 3, strides = 2, padding = 'same')(inputs)
  c = tf.keras.layers.concatenate([u, conv_output])
  c = tf.keras.layers.Dropout(0.3)(c)
  c = conv2d_block(c, n_filters, kernel_size=3)

  return c

def unet_vgg16(input_shape):

    output_channels = 2
    
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("input_1").output
    s2 = vgg16.get_layer("block1_pool").output
    s3 = vgg16.get_layer("block3_conv3").output
    s4 = vgg16.get_layer("block4_conv3").output
    s5 = vgg16.get_layer("block5_conv3").output

    """ Bridge """
    b1 = bottleneck(s5)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    outputs = Conv2D(output_channels, (1, 1),padding="same", activation='softmax')(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model