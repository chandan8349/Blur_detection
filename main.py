import torch
from utils.feature_extractor import FeatureExtractor
from utils.data_loader import TestDataset
from torch.utils.data import DataLoader
from app import app
from flask import flash, request, redirect, render_template
import io
from PIL import Image
import base64
from Helpers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def testing_on_dataset(trained_models, dataset):

    blured = is_image_blurry(trained_models, dataset, threshold=0.6)

    return blured


def is_image_blurry(trained_models, img, threshold=0.6):
    feature_extractor = FeatureExtractor()
    accumulator = []

    # Resize the image by the down sampling factor
    feature_extractor.resize_image(img, np.shape(img)[0], np.shape(img)[1])

    # compute the image ROI using local entropy filter
    feature_extractor.compute_roi()

    # extract the blur features using DCT transform coefficients
    extracted_features = feature_extractor.extract_feature()
    extracted_features = np.array(extracted_features)

    if len(extracted_features) == 0:
        return True
    test_data_loader = DataLoader(TestDataset(extracted_features), batch_size=1, shuffle=False)

    # trained_model.test()
    for batch_num, input_data in enumerate(test_data_loader):
        x = input_data
        x = x.to(device).float()
        output = trained_models(x)
        _, predicted_label = torch.max(output, 1)
        accumulator.append(predicted_label.item())

    blured = np.mean(accumulator) <= threshold
    return blured


def blur(image):
    trained_model = torch.load('./trained_model/trained_model')
    trained_model = trained_model['model_state']

    motion_blur_images = testing_on_dataset(trained_model, image)
    return motion_blur_images


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    images = []
    for file in request.files.getlist("file[]"):
        print("***************************")
        print("image: ", file)
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_str = file.read()
            np_img = np.frombuffer(file_str, np.uint8)
            image = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
            image = Helpers.resize(image, height=500)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            result = "Not Blurry"

            if blur(gray):
                result = "Blurry"

            sharpness_value = "{:.0f}".format(fm)
            message = [result, sharpness_value]

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            file_object = io.BytesIO()
            img = Image.fromarray(Helpers.resize(img, width=500))
            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64," + base64.b64encode(file_object.getvalue()).decode('ascii')
            images.append([message, base64img])

    print("images:", len(images))
    return render_template('upload.html', images=images)


if __name__ == "__main__":
    app.run(debug=True)
