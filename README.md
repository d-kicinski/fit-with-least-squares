# fit-with-least-squares
Click on README.ipynb to see results.

Small project made for Modeling and Identification class. I was given numerical data taken from some SISO(single input single output) object and was asked to create regression model. The main difficulty was to cross-validate the parameters of the model: the degree of non-linearity and degree of dynamic.

The solving strategy was as follows:
 - divide dataset into learning and validation subsets
 - visualise and explore dataset
 - try out the simplest non-dynamic linear and non-linear models to see how it behaves
 - try different dynamic models and add some non-linearities
 - cross-validate model to get best accuracy on validation dataset


Originally it was written in MATLAB but i've rewritten it in Python(numpy + seaborn + pandas + matplotlib) for learning purposes so there's no real raport from experiments but best model can be find in README.ipynb


