## Topic Modeling Script ReadMe

These scripts were developed to utilize the Gensim ldaMulticore library, a parallelized library for completing topic modeling using parallel computing. In this case, they were written to run on the University of Iowa's Argon High Performance Computer (HPC) Cluter. 

The topic modeling scripts are designed to function in tandem with the MulticoreImplementation.py file pulling the parameters for each topic modeling run from the MulticoreInput.py file. These inputs include the name and description of the particular run (to be included in the documentation that is generated each time the MulticoreImplementation Script is run) along with the HathiTrust IDs of the files that the topic model is to by trained on, and the hyperparameters for the topic model (for more information on the specific hyperparameters, see the Gensim ldaMulticore documentation linked below).

Information about the Gensim ldaMulticore library can be found here: https://radimrehurek.com/gensim/models/ldamulticore.html
Information about the Argon High Performance Computing Cluster can be found here: https://wiki.uiowa.edu/display/hpcdocs/Argon+Cluster
