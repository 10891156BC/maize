{
 "cells": [
  
   "cell_type": "code",
   "execution_count": 12,
   "id": "1504ffb4-0f17-41e4-9b76-d09a09fc7581",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pickle\n",
    "import os\n",
    "import sys\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers, models\n",
    "from sklearn.preprocessing import LabelBinarizer\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.preprocessing import image\n",
    "from tensorflow.keras.preprocessing.image import img_to_array\n",
    "import matplotlib.pyplot as plt\n",
    "from keras.applications import ResNet50 \n",
    "from keras.layers import GlobalAveragePooling2D \n",
    "from keras.models import Sequential \n",
    "from keras.layers import Conv2D,MaxPooling2D, Flatten, Dense, Dropout\n",
    "\n",
    "# Define model architecture\n",
    "tf.keras.backend.set_floatx('float32')\n",
    "\n",
    "import sklearn\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8e32d84e-fcb5-4d5f-a15b-ec2a30d369bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "BATCH_SIZE = 32\n",
    "EPOCHS = 50\n",
    "IMAGE_SIZE = 256\n",
    "default_image_size = (IMAGE_SIZE, IMAGE_SIZE)\n",
    "image_size = 0\n",
    "data_dir = \"/code/maize_new\"\n",
    "CHANNELS = 3\n",
    "class_names = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']\n",
    "train_dr = os.path.join(data_dir, 'train')\n",
    "test_dr = os.path.join(data_dir, 'test')\n",
    "val_dr = os.path.join(data_dir, 'val')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5b14c2ab-66db-45d4-8806-c344cfd36625",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Conv2D(32, (3, 3), padding=\"same\", activation='relu', \n",
    "                 input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(64, (3, 3), padding=\"same\", activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(128, (3, 3), activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(512, (3, 3),padding=\"same\", activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Conv2D(512, (3, 3),padding=\"same\", activation='relu'))\n",
    "model.add(MaxPooling2D((2, 2)))\n",
    "model.add(Flatten())\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(1024, activation='relu'))\n",
    "model.add(Dense(4, activation='softmax'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0419a0ec-4d34-4acd-a318-d186192fd69b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='Adam',\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "724b6862-1fb5-4081-aee3-aae69b881fb9",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "                                    rescale=1./255,\n",
    "                                    rotation_range=45,\n",
    "                                    width_shift_range=0.2,\n",
    "                                    height_shift_range=0.2,\n",
    "                                    shear_range=0.2,\n",
    "                                    zoom_range=0.2,\n",
    "                                    horizontal_flip=True,\n",
    "                                    vertical_flip=True,\n",
    "                                    fill_mode='nearest'\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f1136cfb-708f-4e6f-a3e7-c13ac4b52522",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 3348 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = train_datagen.flow_from_directory(\n",
    "        train_dr,\n",
    "        target_size=(IMAGE_SIZE, IMAGE_SIZE),\n",
    "        batch_size=BATCH_SIZE,\n",
    "        class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "37309e4a-0d1d-4044-bdfb-bf32cd8cb175",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_datagen = ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "8890355c-65c5-4a23-a7e9-ab561a0d2cd7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 423 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "test_generator = test_datagen.flow_from_directory(\n",
    "        test_dr,\n",
    "        target_size=(IMAGE_SIZE, IMAGE_SIZE),\n",
    "        batch_size=BATCH_SIZE,\n",
    "        class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "63e82ec2-6760-4e25-9f32-19e41ec53427",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 417 images belonging to 4 classes.\n"
     ]
    }
   ],
   "source": [
    "validation_generator = test_datagen.flow_from_directory(\n",
    "        val_dr,\n",
    "        target_size=(IMAGE_SIZE, IMAGE_SIZE),\n",
    "        batch_size=BATCH_SIZE,\n",
    "        class_mode='categorical')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56f5c6c3-4edd-4b4a-8191-ac9192a3c905",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\MAXWELL\\AppData\\Local\\Temp\\ipykernel_10004\\224633089.py:1: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
      "  model.fit_generator(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "62/62 [==============================] - ETA: 0s - loss: 1.1143 - accuracy: 0.4753WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 25 batches). You may need to use the repeat() function when building your dataset.\n",
      "62/62 [==============================] - 147s 2s/step - loss: 1.1143 - accuracy: 0.4753 - val_loss: 0.9197 - val_accuracy: 0.5180\n",
      "Epoch 2/50\n",
      "62/62 [==============================] - 131s 2s/step - loss: 0.8485 - accuracy: 0.6406\n",
      "Epoch 3/50\n",
      "62/62 [==============================] - 130s 2s/step - loss: 0.8177 - accuracy: 0.6962\n",
      "Epoch 4/50\n",
      "62/62 [==============================] - 127s 2s/step - loss: 0.6534 - accuracy: 0.7591\n",
      "Epoch 5/50\n",
      "62/62 [==============================] - 126s 2s/step - loss: 0.5855 - accuracy: 0.7754\n",
      "Epoch 6/50\n",
      "62/62 [==============================] - 127s 2s/step - loss: 0.5185 - accuracy: 0.7923\n",
      "Epoch 7/50\n",
      "62/62 [==============================] - 126s 2s/step - loss: 0.4509 - accuracy: 0.8201\n",
      "Epoch 8/50\n",
      "62/62 [==============================] - 126s 2s/step - loss: 0.4615 - accuracy: 0.8170\n",
      "Epoch 9/50\n",
      "62/62 [==============================] - 124s 2s/step - loss: 0.4139 - accuracy: 0.8337\n",
      "Epoch 10/50\n",
      "62/62 [==============================] - 124s 2s/step - loss: 0.4011 - accuracy: 0.8332\n",
      "Epoch 11/50\n",
      "62/62 [==============================] - 122s 2s/step - loss: 0.3880 - accuracy: 0.8367\n",
      "Epoch 12/50\n",
      "62/62 [==============================] - 120s 2s/step - loss: 0.3792 - accuracy: 0.8499\n",
      "Epoch 13/50\n",
      "62/62 [==============================] - 121s 2s/step - loss: 0.3589 - accuracy: 0.8514\n",
      "Epoch 14/50\n",
      "62/62 [==============================] - 122s 2s/step - loss: 0.3702 - accuracy: 0.8494\n",
      "Epoch 15/50\n",
      "62/62 [==============================] - ETA: 0s - loss: 0.3050 - accuracy: 0.8803"
     ]
    }
   ],
   "source": [
    "model.fit_generator(\n",
    "        train_generator,\n",
    "        steps_per_epoch=2000 // BATCH_SIZE,\n",
    "        epochs=50,\n",
    "        validation_data=validation_generator,\n",
    "        validation_steps=800 // BATCH_SIZE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4aca2550-36b2-4d9a-83e5-87f922f3ce9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.evaluate_generator(test_generator)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0d8d6e3-36a6-4e19-91fa-f3b5b15daaa1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Plot training & validation accuracy values\n",
    "plt.plot(history.history['accuracy'])\n",
    "plt.plot(history.history['val_accuracy'])\n",
    "plt.title('Model accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['Train', 'Test'], loc='upper left')\n",
    "plt.show()\n",
    "\n",
    "# Plot training & validation loss values\n",
    "plt.plot(history.history['loss'])\n",
    "plt.plot(history.history['val_loss'])\n",
    "plt.title('Model loss')\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epoch')\n",
    "plt.legend(['Train', 'Test'], loc='upper left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5604a94c-a122-4bd1-af29-b7316e6af64f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  },
  "toc-autonumbering": false,
  "toc-showcode": false,
  "toc-showmarkdowntxt": true,
  "toc-showtags": true
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
