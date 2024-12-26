# ML-Workshop
This workshop will go over the very common first example of handwritten digit recognition using the MNIST database and PyTorch as the machine learning framework of choice. 
This README will contain a summarized version of the workshop with step-by-step instructions and explanations. Some python knowledge is assumed, including how to create a python file, installing packages and basic syntax. 

## Creating an Environment
First we create a virtual environment. While it's not necessary, it helps keeps projects seperate and portable. To create a virtual environment, open a terminal in the folder of your choice. 
```bash
python3 -m venv env
```
For those unfamiliar with bash, python3 is the name of the executable. -m is a flag, which tells the python3 executable to look in its modules for a module named venv. Venv takes a user defined argument, 'env'. 'env' is the name of the folder which holds the virtual environment. Next the virtual environment should be activated. To do that you can do the following. 
On a UNIX-like system (MacOS or Linux)
```bash
source env/bin/activate
```
On windows
```bash
env/Scripts/activate
```
If everything went correctly, you should notice the primary prompt string, the characters to the left in your terminal, has changed. 

## Installing packages
There are only two packages that need to be installed for this project: Torch and Torchvision. Installation instructions are available here: [PyTorch](https://pytorch.org/get-started/locally/). They vary as hardware changes: mostly due to the presence of CUDA for nvidia gpu parallelization. The basic installations using pip will be:
```bash
#Please view the resources above before installation. 
pip install torch torchvision
```
To check if installations were successful, enter `pip list`. There will be more than just two packages. 

## About MNIST
For the purposes of this workshop, there are a few key details and assumptions that ought to be established. Modified National Institute of Standards and Technology (MNIST) Database is a database of handwritten digits, often used to teach and develop machine learning algorithms. The dataset is split into two parts. A training set, and a testing set. The current dataset is a greyscale image, with each pixel having a corresponding color value between 0-255. Each image is 28x28 pixels. 




