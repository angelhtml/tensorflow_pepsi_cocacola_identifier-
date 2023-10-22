# tensorflow pepsi cocacola identifier

<p align="center">
  <img src="https://iili.io/JKzkf1V.jpg" alt="ml" />
</p>

<h1>How to use ⁉</h1>
<p>First of all download the data set from <a href="https://www.kaggle.com/datasets/die9origephit/pepsi-and-cocacola-images/data">here</a></p>
<p>Then extract the zip inside the input folder</p>

<h3>Train your dataset</h3>
```cmd
python train.py
```

<h3>Test your model</h3>
enter your test image here

```py
image = tf.keras.utils.load_img("input/test/pepsi/18.jpg",target_size=(25,25))
```

```cmd
python test.py
```
