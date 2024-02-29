#Ship Segmentation using Deep Learning
This repository contains a solution for the task of segmenting ships in photos using deep learning techniques. Below is a detailed description of the solution approach and implementation.

Solution Description
Data Analysis:
An initial analysis of the data revealed that approximately 87% of the images did not contain ships. To expedite the training process, images without ships were excluded from the training dataset.
Model Implementation:
Was implemented UNet with 300k parameteres. Size of it was reduced from traditional net so that training would take less time and show good results. 

Dice Score and IoU: Dice score and Intersection over Union (IoU) metrics were implemented using TensorFlow and Keras. These metrics are commonly used to evaluate the performance of segmentation models.

The model was trained using the curated dataset. After 3 epochs of training, the Dice score exceeded 0.99, indicating excellent segmentation performance.
For more details look in the notebook.
