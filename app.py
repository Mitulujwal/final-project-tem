from flask import Flask, render_template, request, redirect, url_for
import requests
from PIL import Image
from io import BytesIO
from PIL import *
from PIL import ImageDraw

app = Flask(__name__)
class AddLoss(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.lossfn = keras.backend.ctc_batch_cost
    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.lossfn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

# building our model using the functional keras api
def my_model():
    # CNN
    inputt = layers.Input(shape=(img_width, img_height, 1), name="img", dtype="float32" )
    label = layers.Input(name="label", shape=(None,), dtype="float32")
    x = layers.Conv2D(64,(3, 3),activation="relu",padding="same")(inputt)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128,(3, 3),activation="relu",padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3,3), activation="relu", padding="same")(x)
    
    x = layers.Reshape(target_shape=((img_width // 4), (img_height // 4) * 256))(x)
    x = layers.Dense(256, activation="relu", name="dense_layer1")(x)

    # GRU is faster than LSTM because it only has 2 gates, while LSTM has 3 gates
    x = layers.Bidirectional(layers.GRU(256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    # output layer, number of units = the total number of possible characters in car plates + one character for unknown charcters
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense_layer2")(x)

    output = AddLoss(name="loss")(label, x)
    # build keras Model
    model = tf.keras.models.Model(inputs=[inputt, label], outputs=output)

    return model

model = my_model()
model.compile(optimizer="Adam")
model.summary()

epochs = 30

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs)

#   
pred_model = tf.keras.models.Model(model.get_layer(name="img").input, model.get_layer(name="dense_layer2").output)

# predection function, loads image, reshape it to the correct shape for the model, sends it to the model, decode the prediction and return it
def predict(im_file):
  img = load_sample_to_memory(im_file, "label")
  img = img["image"].numpy()
  img = img.reshape((-1,img_width,img_height,1))
  pred = pred_model.predict(img)
  decoded_pred = decode_predictions(pred)
  return decoded_pred
def detect_license_plate(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text
#  visualize results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["img"]
    batch_labels = batch["label"]

    preds = pred_model.predict(batch_images)
    pred_texts = decode_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        result = detect_license_plate(image)
        if 'results' in result and len(result['results']) > 0:
            plate = result['results'][0]
            license_plate = plate['plate'].upper()
            print(type(license_plate))
            box = plate['box']
            image = Image.open(image)
            image_with_box = image.copy()
            image_with_box=image_with_box.convert('RGB')
            draw = ImageDraw.Draw(image_with_box)
            draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline='red', width=3)
            image_with_box.save('static/license_plate.jpg')
            return render_template('result.html', license_plate=license_plate)
        else:
            return render_template('index.html', error='No license plate detected.')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
