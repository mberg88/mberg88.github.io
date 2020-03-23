# copy of conv2D-mnist
# https://towardsdatascience.com/deploying-a-simple-machine-learning-model-into-a-webapp-using-tensorflow-js-3609c297fb04
# works on python 3.7
# use GPU
python37 webapp_main.py

# works on python 3.6
# https://stackoverflow.com/a/55734209
python36 -m pip install tensorflowjs

# tensorflowjs_converter cannot run because of imp
# install the latest Microsoft Visual C++ redistributable: read
https://github.com/tensorflow/tensorflow/issues/35749#issuecomment-588425399
https://github.com/tensorflow/tensorflow/issues/35749#issuecomment-596627346

# convert to JSON
tensorflowjs_converter --input_format keras results/model.h5 results/model

# folder results/model contains group1-shard1of1.bin and model.json
# model.json: the dataflow graph and weight manifest file
# group1-shard\*of\*: collection of binary weight files

# save HTML file tfjs_main.html to folder results
# change it if needed

# test things locally
python37 -m http.server 8080
# go to http://localhost:8080/tfjs_main.html to draw numbers

# build a website
# read how-to here:
# https://pages.github.com/
# go to https://my-username.github.io
# in my case, go to mberg88.github.io/results/tfjs_main.html

