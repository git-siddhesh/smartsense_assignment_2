# smartsense_assignment_2


# Directory structure 

- data - contains the folder to the images of different people
- dataset - contrains two folder train and test  created using create_dataset.py
    - train - contains the images of different people
    - test - contains the images of different people
- model - contains the model.h5 files

PYTHON NOTEBOOKS and FILES
- `EDA.ipynb`  - Notebook containing all the EDAs
- `create_dataset.py` - this file is used to create the dataset {train and test}
- `train.py` - this file is used to train the model
- `results.csv` - this file contains the results of the test data on different model



# How to run the code

### Application

To run the application, run the `app.py` file

```bash
app.py --model vgg --blocks 16 --usename sidd
```

> Command line arguments:

```python
model: str = 'vgg'
blocks: int = 16
username: str = 'sidd'
clicks: int = 5
```


> NOTE: when this application runs; It will open you camera and try to capture some images of yous `dafault = 5`
---
iye chai
##### How to capture images

- Give a nice smile and ready pose :)
- Press `space` to capture the image
- Press `ESC` to stop or it will capture 5 images by default


> NOTE: The images will be saved in the `data` folder with the name `username`

---

### Dataset_creation

run the create_dataset.py file to create the dataset
```bash
python create_dataset.py --data data
```

This will create a `dataset` folder containing two subfolders.

---

### Training the model

Then run the train.py file to train the model
```python
parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--model', type=str, default='vgg', help='model name')
parser.add_argument('--blocks', type=int, default=1, help='number of blocks 1,3,5,..')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--batch', type=int, default=5, help='batch size')
parser.add_argument('--imgsize', type=int, default=128, help='image size')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--augment', type=bool, default=False, help='data augmentation')
```

```bash
python train.py --model vgg --blocks 16 --epochs 10 --batch 32 
```

This will create a model folder containing the model.h5 file: for now : `vgg_16.h5`

And save the evaluation results in the `results2.csv` file

---




