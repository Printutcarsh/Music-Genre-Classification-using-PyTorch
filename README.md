# Music-Genre-Classification-using-PyTorch
<p align="center">
<img src='https://themusicda.com/wp-content/uploads/2018/01/da4d0444-57e3-4064-8044-1eb0d443bf80_560_420.jpg'></img>
</p>
<hr>
<b> Dataset Link :- </b> https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br>
<b> About the data :- </b> Genre original folder(only this required) - It is a collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds).<br>
# Implementation Guide
<ul>
  <li> First of all, we have to do some data preprocessing and extract some useful information from our music data so that we can use it for training our model. For that run prepare_dataset.py.</li>
  <li> Next there are 2 custom model(one is cnn based and the other is rnn based) built using PyTorch and it is trained on the preprocessed data. 
    For this run audio_cnn_pytorch for cnn based model or audio_rnn_pytorch for rnn based model.</li>
</ul>
