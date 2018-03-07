import json
import urllib.request
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# main
# Load local city data
df = pd.read_csv('../data/world-cities_csv.csv')
df_jp = df[df['country'] == "Japan"]
# Add Lat Lng
latlng_city = lookup_latlng(df_jp, '')
# Add BigCity flg
flg_latlng_city = big_city_flagger(latlng_city)
# Get satellite data
get_ggl_image_sep(flg_latlng_city)


def lookup_latlng(city_df, api_key=''):
    """
    Convert address name to latitude and longitude
    @ param city_df: name is city name, subordinary is prefecture name (https://datahub.io/core/world-cities)
    @ param api_key: APY Key for Google Map API
    @ retrun city_df: lat, lng added city data
    """
    #city_df = city_df.head()
    city_l = city_df['name'] + ',' + city_df['subcountry']
    lat = []
    lng = []
    ggl_url = "https://maps.googleapis.com/maps/api/geocode/json?address="
    api_key = "AIzaSyCrUUNVdkMhCcQ17u-WBjgDTQz2wDi5ng4"
    # GEOCODING Lookup, by Google Maps API
    for w in city_l:
        print(w)
        url = ggl_url + w + "&key=" + api_key
        response = urllib.request.urlopen(url)
        content = json.loads(response.read().decode('utf8'))
        if content['status'] == 'OK':
            lat_tmp = content['results'][0]['geometry']['location']['lat']
            lng_tmp = content['results'][0]['geometry']['location']['lng']
        else:
            lat_tmp = 0
            lng_tmp = 0
        lat.append(lat_tmp)
        lng.append(lng_tmp)

    city_df['lat'] = lat
    city_df['lng'] = lng
    city_df = city_df[city_df.lat != 0]
    #print(city_df.head())
    #pd.DataFrame.to_csv(city_df, "../data/CityData_latlng.csv")
    return city_df

def big_city_flagger(city_df, flg_pass='../data/BigCityList.csv'):
    """
    Add flag of big city
    @ param city_df: lat, lng added city data
    @ return city_df: city data having big city flag
    """
    #flg_city = pd.read_csv(flg_pass)['big_city']
    flg_city_l = ['Tokyo', 'Kyoto', 'Osaka', 'Nagoya', 'Yokohama', 'Kobe', 'Kitakyushu', 'Kawasaki', 'Fukuoka',
                  'Hiroshima', 'Sendai', 'Chiba', 'Saitama', 'Shizuoka', 'Sakai', 'Niigata', 'Hamamatsu', 'Okayama',
                  'Sagamihara', 'Kumamoto']
    flg_l = []
    cnt = 0
    for city in city_df['name']:
        if city in flg_city_l:
            flg_l.append(1)
            cnt += cnt
        else:
            flg_l.append(0)
    city_df['flg'] = flg_l
    print(cnt == len(flg_city_l)) # 未対応：Sakai, Kawasakiが複数市でヒット。Sendaiはデータに重複（データ重複は後工程的に問題ない）
    return city_df

def get_ggl_image_sep(city_df, zoom=16, size=320, img_dir='../imgs/'):
    """
    Get google satellite image via Google Map API
    @ param city_df: df including city name and latlng
    @ param zoom: zoom level
    @ param size: image size
    @ param img_dir: directory pass for image folder
    """
    # Map Configue
    lat = city_df['lat']
    lng = city_df['lng']
    city_name = city_df['flg'].astype(str) + '_' + city_df['name'] + '_' + city_df['subcountry']
    im_type = "satellite"
    api_key = "AIzaSyBG0yssmOa53O3uJf8HDD0ikoD3SioeD5M" # <- confidential
    ggl_url = "https://maps.googleapis.com/maps/api/staticmap"
    cnt = 0
    # Scraping google map
    for i in city_name.index:
        # Issue the image url
        img_url = ggl_url + "?center=" + str(lat[i]) + "," + str(lng[i]) + "&zoom=" + str(zoom) + "&size=" + str(size) + "x" + str(size) + "&maptype=" + im_type + "&key=" + api_key
        print(city_name[i])
        print(img_url)
        # Save images
        if city_name[i][0] == '1' and i <= int(len(city_name) * 0.8):
            urllib.request.urlretrieve(img_url, img_dir  + 'train/bigCity/' + city_name[i] + ".jpg")
        elif city_name[i][0] == '0' and cnt <= int(len(city_name) * 0.8):
            urllib.request.urlretrieve(img_url, img_dir + 'train/normCity/' + city_name[i] + ".jpg")
        elif city_name[i][0] == '1' and cnt > int(len(city_name) * 0.8):
            urllib.request.urlretrieve(img_url, img_dir + 'test/bigCity/' + city_name[i] + ".jpg")
        elif city_name[i][0] == '0' and cnt > int(len(city_name) * 0.8):
            urllib.request.urlretrieve(img_url, img_dir + 'test/normCity/' + city_name[i] + ".jpg")
        cnt += 1
    print(str(cnt) + " images are stored at " + img_dir)

# model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.image as mpimg
from scipy.misc import imresize
import keras.backend as K

# Generate Train and Validation data
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    '../imgs/train',
    target_size=(320, 320),
    batch_size=32,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    '../imgs/test',
    target_size=(320, 320),
    batch_size=32,
    class_mode='binary')

# Load Inception v3 without last layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Define last layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid", kernel_regularizer=l2(.0005))(x)

model = Model(inputs=base_model.input, outputs=predictions)

# base_mode and lweights are not updated
for layer in base_model.layers:
    layer.trainable = False

opt = SGD(lr=.01, momentum=.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model.log')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                  patience=5, min_lr=0.001)

history = model.fit_generator(train_generator,
                    steps_per_epoch=2000,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=800,
                    verbose=1,
                    callbacks=[reduce_lr, csv_logger, checkpointer])
