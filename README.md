# Tensorflow-quickstart-offline
Base on [Tensorflow official quickstart][tensorflow-docs], but to load train data locally(offline) instead of downloading online, avoid blocking by GFW in China or others.

# For whom
Blocked by GFW(like China) can't download `tensorflow.keras.datasets.*` by calling `load_data()` methods.

# Method
<table class="tg">
  <tr>
    <th class="tg-88nc">keras.datasets</th>
    <th class="tg-88nc">local_datasets</th>
  </tr>
  <tr>
    <td class="tg-0pky">imdb.get_word_index</td>
    <td class="tg-0pky">get_word_index</td>
  </tr>
  <tr>
    <td class="tg-0pky">imdb.load_data</td>
    <td class="tg-0pky">load_data_imdb</td>
  </tr>
  <tr>
    <td class="tg-0pky">mnist.load_data</td>
    <td class="tg-0pky">load_data_mnist</td>
  </tr>
  <tr>
    <td class="tg-0pky">fashion_mnist.load_data</td>
    <td class="tg-0pky">load_data_fashion_mnist</td>
  </tr>
  <tr>
    <td class="tg-0pky">boston_housing.load_data</td>
    <td class="tg-0pky">load_data_boston_housing</td>
  </tr>
</table>


# Usage

## For quick start

Open `notebook` in `official_quickstart`.

## For your examples project

copy `utils` directory to any project.

### python
```python
from utils.local_datasets import load_data_fashion_mnist

(train_images, train_labels), (test_images, test_labels) = load_data_fashion_mnist()
```

### jupyter notebook
```jupyter
import os
import sys
# add utils path to sys.path in order to import
# PS: be aware of the absolute path of utils
module_path = os.path.abspath(os.path.join('utils'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from local_datasets import load_data_fashion_mnist

(train_images, train_labels), (test_images, test_labels) = load_data_fashion_mnist()
```


[tensorflow-docs]: https://github.com/tensorflow/docs
