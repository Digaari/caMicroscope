# caMicroscope - Cancer Region of Interest Extraction and Machine Learning

### Mentors
  **Insiyah Hajoori** and **Ryan Birmingham**

### Code Challenge:
Using a machine learning toolkit of your choice, create a tool which identifies objects in the image, then returns positions in pixels corresponding to bounding boxes of a user-selected class of object in the image. For example, given an image with both cats and dogs, return bounding boxes for only cats.

### Prerequisites

* Python **3.7.3 or above**
* Node **v12.16.1 or above**
* You will need the weights. Get them from: https://drive.google.com/drive/folders/1lNKCbOM7VSRLIvHB2jh7XQAThPRHmpt_?usp=sharing

### Procedure

1. Clone this repository.
2. Download the weights from the link above.
3. Put the downloaded weights into the following folder:
```
 gsoc_model
```
4. Your ```gsoc_model``` folder will now have the following contents:
```
 coco.names 
 gsoc.cfg
 gsoc.weights
 gsoc.txt
 
```
5. On the terminal:
```
 cd caMicroscope
 pip install virtualenv
 source env/Scripts/activate
 pip install -r requirements.txt
```
6. Install Pyrebase4 4.3.0
```
pip install Pyrebase4
```
7. Run the Flask app
```
python app.py
```
8. Choose a file from ```Images```
9. Your File has been detected. Thanks!

## Author

[Hrishabh Digaari](https://www.linkedin.com/in/hrishabh-d-35aa60127/) - LinkedIn

## Acknowledgments

* https://arxiv.org/pdf/1506.02640.pdf
* https://arxiv.org/pdf/1804.02767.pdf
* https://flask.palletsprojects.com/
* https://firebase.google.com/docs/web/setup
