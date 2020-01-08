# ICRAR-Internship
Code regarding data filtering using a machine learning approach.

AstroQuest is an online citizen science project undertaken by the International Centre for Radio Astronomy Research, WA.

It takes a crowd-sourcing approach and asks regular folk to get involved and help the scientists with their research work. The citizen scientists are inspecting images of galaxies and helping Australian astronomers with their research into how galaxies grow and evolve

Unfortunately, some of the data is unusable because of:
1.	Data errors
2.	User errors
3.	Malicious users 

This project is to develop a Machine Learning system to scan all the images (70,000+) looking for bad images and removing them from the dataset.

NOTE:
This repo is still being constructed so parts are currently incomplete or unorganised

## Method
The Machine Learning system will be made up of multiple trained neural networks that look at the data for different errors. The base program of the system aims to remove 3 types of errors. The networks all contain convolutional neural network parts to make the image processing more effective, and some use transfer learning to take advantage of much large pretrained models.

Firstly, a neural network is being used to remove a bug in the data (nicknamed TopCorner) which moves the centred drawing to top left corner and fill in the rest of the screen.

The second network is there to remove user generated masks where the user has scribbled on the drawing. In order to generate bad data images of hand-drawn scribbles have been sourced from the Quick! Draw data (https://quickdraw.withgoogle.com/data) and overlayed over good images. This allows for large amounts of data with scribbles to be generated to train the model. This model could potentially be expanded to all of the Quick! Draw data so it is able to pick a much larger range of drawings as well as scribbles.

The third and final (for now) model removes masks with text in them. The data for this is sourced from EMNIST which contains many hand-written letters and digits. These will be overlayed over good images similar to the second network to simulate malicious user input, then used to train the model.

More networks may be created in later iterations.

## Results
The project has currently not yielded results yet and is still in development.
