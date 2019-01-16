# tsp
Tertiary Structure Prediction

## databases
The databases folder contains the fasta files used to construct our custom BLAST databases. It of course also contains those database files.

## models
The models folder contains saved models from various stages of the pipeline. Their names are descriptive. Note: Some models were too large, and they are on the shared Google Drive.

## notebooks
The notebooks folder contains several ipynb which we used to develop our code. No guarantee as to how updated each file is. All code used to produce results is found in src. A notable exception is plot.ipynb, which we used to produce some plots.

## predictions
~Our predictions from various stages of the pipeline. Again, the names are descriptive.~ Files were too large; we uploaded to the shared Google Drive.

## src

auto.py: contains the code for using Auto Scikit Learn to train two regressors, to predict the mean and standard deviation of sequence's distance matrix, respectively.

dcgan.py: Our best attempt at implementing DCGAN. This was the code used to produce the generator and discriminator models in the models folder.

dcgan2.0.py: Another GAN attempt.

helperfunctions.py: Some functions for setting up BLAST databases and performing matches.

model1.py: An older attempt at an end-to-end model. No predictions produced from this model.

scale.py: Uses the mean and standard deviation predictions of the regressors from auto.py to scale distance matrix predictions.

seq2mat_st.py: The improved end2end model with the overlay of matched segments from BLAST searches.

seq2mat.py: Our older end2end model.
