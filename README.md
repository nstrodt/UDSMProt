# UDSMProt: Universal Deep Sequence Models for Protein Classification
**UDSMProt** is an algorithm for the classification of proteins based on the sequence of amino acids alone. Its key component is a self-supervised pretraining step based on a language modeling task. The model is then subsequently finetuned to specific classification tasks. In our paper we considered enzyme class classification, gene ontology prediction and remote homology detection showcasing the excellent performance of **UDSMProt**.

For a detailed description of technical details and experimental results, please refer to our paper:

[Universal Deep Sequence Models for Protein Classification](https://doi.org/10.1101/704874)

Nils Strodthoff, Patrick Wagner, Markus Wenzel, and Wojciech Samek

bioRxiv preprint 2019

    @article{Strodthoff:2019UDSMProt,
	author = {Strodthoff, Nils and Wagner, Patrick and Wenzel, Markus and Samek, Wojciech},
	title = {{UDSMProt: Universal Deep Sequence Models for Protein Classification}},
	elocation-id = {704874},
	year = {2019},
	doi = {10.1101/704874},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
	}

This is the accompanying code repository, where we also provide links to [pretrained language models](https://datacloud.hhi.fraunhofer.de/nextcloud/s/9R8mWzDSYWdQdjd). 

Also have a look at [**USMPep**](https://github.com/nstrodt/USMPep):[Universal Sequence Models for Major Histocompatibility Complex Binding Affinity Prediction](https://doi.org/10.1101/816546) that builds on the same framework.


## Dependencies
for training/evaluation: `pytorch` `fastai` `fire` 

for dataset creation: `numpy` `pandas` `scikit-learn` `biopython` `sentencepiece` `lxml`

## Installation
We recommend using conda as Python package and environment manager.
Either install the environment using the provided `proteomics.yml` by running `conda env create -f proteomics.yml` or follow the steps below:
1. Create conda environment: `conda create -n proteomics` and `conda activate proteomics`
2. Install pytorch: `conda install pytorch -c pytorch`
3. Install fastai: `conda install -c fastai fastai=1.0.52`
4. Install fire: `conda install fire -c conda-forge`
5. Install scikit-learn: `conda install scikit-learn`
6. Install Biopython: `conda install biopython -c conda-forge`
7. Install sentencepiece: `pip install sentencepiece`
8. Install lxml: `conda install lxml`

Optionally (for support of threshold 0.4 clusters) install [cd-hit](`https://github.com/weizhongli/cdhit`) and add `cd-hit` to the default searchpath.


## Data
### Swiss-Prot and UniRef
* Download and extract the desired Swiss-Prot release (by default we use 2017_03) from the [UniProt ftp server](ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2017_03/knowledgebase/uniprot_sprot-only2017_03.tar.gz). Save the contained `uniprot_sprot.xml` as `uniprot_sprot_YEAR_MONTH.xml` in the `./data` directory 
* Download and extract the desired UniRef release (by default we use 2017_03) from the [UniProt ftp server](ftp://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2017_03/uniref/uniref2017_03.tar.gz). Save the contained `uniref50.xml` as `uniref50_YEAR_MONTH.xml` in the `./data` directory. As an alternative and for full reproducibility, we also provide [pickled cluster files](https://datacloud.hhi.fraunhofer.de/nextcloud/s/9R8mWzDSYWdQdjd) `cdhit04_uniprot_sprot_2016_07.pkl` and `uniref50_2017_03_uniprot_sprot_2017_03.pkl` to be placed under `./tmp_data` that avoid downloading the full UniRef file or running cd-hit.
* Or just call our provided script `./download_swissprot_uniref.sh 2017 03` which manages everything for you.

### EC prediction
* Preprocessed versions of the [DEEPre](http://www.cbrc.kaust.edu.sa/DEEPre/dataset.html) and [ECPred](https://github.com/cansyl/ECPred) datasets are already contained in the `./git_data` folder of the repository.
* The custom EC40 and EC50 datasets will be created from Swiss-Prot data directly.
 

### GO prediction
* Download the raw GO prediction data `data-2016.tar.gz` from [DeepGoPlus](http://deepgoplus.bio2vec.net/data/) and extract it into the `./data/deepgoplus_data_2016` folder

### Remote Homology Detection
* Download the [superfamily](`http://www.bioinf.jku.at/software/LSTM_protein/jLSTM_protein/datasets/SCOP167-superfamily.tar.bz2`) and [fold](http://www.bioinf.jku.at/software/LSTM_protein/jLSTM_protein/datasets/SCOP167-fold.tar.bz2) datasets and extract them into the `./data` folder


## Data Preprocessing
* Run the data preparation script 

```shell
cd code 
./create_datasets.sh
```

* The output is structured as follows: 
    * `tok.npy` sequences as list of numerical indices (mapping is provided by `tok_itos.npy`)
    * `label.npy` (if applicable) label as list of numerical indices (mapping is provided by `label_itos.npy`)
    * `train_IDs.npy`/`val_IDs.npy`/`test_IDs.npy` numerical indices identifying training/validation/test set by specifying rows in `tok.npy`
    * `train_IDs_prev.npy`/`val_IDs_prev.npy`/`test_IDs_prev.npy` original non-numerical IDs for all entries that were ever assigned to the respective sets (used to obtain consistent splits for downstream tasks)
    * `ID.npy` original non-numerical IDs for all entries in `tok.npy`
* The approach is easily extendable to further downstream classification or regression tasks. It only requires to implement a corresponding preprocessing method similar to the ones provided for the existing tasks in `preprocessing_proteomics.py`.

## Basic Usage
We provide some basic usage information for the most common tasks:
* Language Model Pretraining (or skip this step and use the [provided pretrained LMs](https://datacloud.hhi.fraunhofer.de/nextcloud/s/9R8mWzDSYWdQdjd) (forward and backward models trained on SwissProt 2017_03))
```shell
cd code
python modelv1.py language_model --epochs=60 --lr=0.01 --working_folder=datasets/lm/lm_sprot_dirty/ --export_preds=False --eval_on_val_test=True
```
* Finetuning for enzyme class classification (here for level 1 and EC50 dataset; assuming the pretrained folder is located at `datasets/lm/lm_sprot_uniref_fwd`)
```shell
cd code
python modelv1.py classification --from_scratch=False --pretrained_folder=datasets/lm/lm_sprot_uniref_fwd --epochs=30 --metrics=["accuracy","macro_f1"] --lr=0.001 --lr_fixed=True --bs=32 --lr_slice_exponent=2.0 --working_folder=datasets/clas_ec/clas_ec_ec50_level1 --export_preds=True --eval_on_val_test=True
```
* Finetuning for gene ontology prediction
```shell
cd code
python modelv1.py classification --from_scratch=False --pretrained_folder=datasets/lm/lm_sprot_uniref_fwd --epochs=30 --lr=0.001 --lr_fixed=True --bs=32 --lin_ftrs=[1024] --lr_slice_exponent=2.0 --metrics=[] --working_folder=datasets/clas_go/clas_go_deepgoplus_2016 --export_preds=True --eval_on_val_test=True
```
* Finetuning for remote homology detection (here for superfamily level and a single dataset)
```shell
cd code
python modelv1.py classification --from_scratch=False --pretrained_folder=datasets/lm/lm_sprot_uniref_fwd --epochs=10 --bs=128 --metrics=["binary_auc","binary_auc50","accuracy"] --early_stopping=binary_auc --bs=64 --lr=0.05 --fit_one_cycle=False --working_folder=datasets/clas_scop/clas_scop0 --export_preds=True --eval_on_val_test=True
```
The output is logged in `logfile.log` in the working directory, the final results are exported for convenience as `result.npy` and individual predictions that can be used for example for ensembling forward and backward models are exported as `preds_valid.npz` and `preds_valid.npz` (in case `export_preds` is set to true).
