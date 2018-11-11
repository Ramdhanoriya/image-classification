Image Classification using tf.data and tf.keras
----------------
This project helps understand the usage of [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data) and [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras) API in image classifcation tasks. 

Requirements
----------------------
* python3
* tensorflow 1.12.0
* sklearn

Command-line arguments
----------------------
* Input Parameters
	* `--data_dir` : Directory contains train images. Expecting a directory structure like below. Sub directory names are the class labels.
  ```javascript
  ├── train
  │   ├── Class_Label_1
  │   │   ├── Class_Label_1_1139.jpg
  │   │   ├── Class_Label_1_1140.jpg
  │   ├── Class_Label_2
  │   │   ├── Class_Label_2_1139.jpg
  │   │   ├── Class_Label_2_1140.jpg
  │   ├── Class_Label_3
  │   │   ├── Class_Label_3_1139.jpg
  │   │   ├── Class_Label_3_1140.jpg
  
  ```
  * `--seed` : Random seed to re-produce train-test split
  * `--batch_size` : Train batch size to fetch from tf.data.Dataset
  * `--no_threads` : Number of threads to run pre-processing and augmentation process with in tf.data.Dataset.map
  * `--mode` : Mode to execute. `train` or `test`
* Train Parameters
  * `--no_epochs` : Number of epochs
  * `--no_class` : Total number of class labels in the training data
  * `--epochs_steps` : Total number of epoch steps per epoch
  * `--val_split_ratio` : Validation split ratio
* Pre-process/Augmentation Parameters
  * `--center_crop` : Whether to perform center crop or not.
  * `--crop_size` : If `--center_crop` enabled , provide `--crop_size`
  * `--resize` : Whether to resize images or not.
  * `--image_size` : If `--resize` enabled , then provide `--image_size`
  * `--flip` : If enabled , images will be flipped horizontally left to right
  * `--transpose` : If enabled , images will be transposed
  * `--random_augment` : If enabled , random hue, saturation, brightness and contrast will be applied to images.
* Output Parameters
  * `--model_dir` : Directory to checkpoint model at each epoch, write model json and model.h5
  * `--log_dir` : Direcory to write logs
* Test Parameters
  * `--test_dir` : Test directory containing test images to evaluvate and predict. Structure should be same as above.
  * `--model_file'` : File path of stored model.h5 file

Training and Validation
----------------------
```javascript

python train.py --resize --image_size 300 --center_crop --crop_size 1000 --epochs_steps 50 --no_epochs 100 --data_dir ../data/train_val --no_class 20 --batch_size 128 --flip --transpose --random_augment --mode train

```
Evaluvate and Predict
----------------------
```javascript

python train.py --resize --image_size 300 --center_crop --crop_size 1000 --test_dir ../data/test/ --model_file ../model/model.h5 --mode test

```
`--center_crop` and `--resize` applied at the training phase has to be replicated for `test` mode with excat `--image_size` and `--crop_size`
