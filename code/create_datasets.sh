mkdir -p datasets
mkdir -p datasets/lm
mkdir -p datasets/clas_ec
mkdir -p datasets/clas_scop
mkdir -p datasets/clas_scop_fold
mkdir -p datasets/clas_go

##################################
#default LMs
##################################
#lm sprot (default=clustered)
python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_uniref --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=False

#lm sprot dirty (disregarding clusters)
python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_dirty --ignore_clusters=True --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=False
#lm sprot dirty subsampling
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_dirty_subsampling633 --ignore_clusters=True --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=False --subsampling_ratio_train=0.633

#cdhit04 dirty
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_dirty2 --cluster_type=cdhit04 --ignore_clusters=True --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=False --source_sprot=../data/uniprot_sprot_2016_07.xml

##################################
#EC uniprot (EC50)
##################################
python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level0 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True
python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True

#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level0_nonredundant --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="uniprot" --sampling_method_train=3 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] 
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_nonredundant --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=3 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] 

##################################
#EC cdhit04 (EC40)
##################################
python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec40_level0 --pretrained_folder=datasets/lm/lm_sprot_dirty --level=0 --include_NoEC=True --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml
python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec40_level1 --pretrained_folder=datasets/lm/lm_sprot_dirty --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec40_level0_nonredundant --pretrained_folder=datasets/lm/lm_sprot_dirty --level=0 --include_NoEC=True --dataset="uniprot" --sampling_method_train=3 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec40_level1_nonredundant --pretrained_folder=datasets/lm/lm_sprot_dirty --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=3 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml

##################################
#EC uniprot subsampling (EC50)
##################################
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling75 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.75
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling50 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.50
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling25 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.25
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling10 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.10
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling05 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.05
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling01 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.01
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_subsampling001 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --subsampling_ratio_train=0.001

##################################
#clean LMs (EC50)
##################################
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_clean_clas_ec_ec50_level0 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --pretrained_folder=datasets/clas_ec/clas_ec_ec50_level0
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_clean_clas_ec_ec50_level1 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --pretrained_folder=datasets/clas_ec/clas_ec_ec50_level1

##################################
#clean LMs (EC40)
##################################
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_clean_clas_ec_ec40_level0 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --pretrained_folder=datasets/clas_ec/clas_ec_ec40_level0 --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_clean_clas_ec_ec40_level1 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --pretrained_folder=datasets/clas_ec/clas_ec_ec40_level1 --cluster_type=cdhit04 --source_sprot=../data/uniprot_sprot_2016_07.xml

##################################
#SCOP superfamily and fold
##################################
#with train val split
for x in {0..101}
do
    python proteomics_preprocessing.py clas_scop --scop_select=$x --split_ratio_train=0.99 --working_folder=datasets/clas_scop/clas_scop$x --pretrained_folder=datasets/lm/lm_sprot_uniref
done

for x in {0..84}
do
    python proteomics_preprocessing.py clas_scop --scop_select=$x --split_ratio_train=0.99 --working_folder=datasets/clas_scop_fold/clas_scop_fold$x --pretrained_folder=datasets/lm/lm_sprot_uniref --foldername=SCOP167-fold
done
#issue with 56 (val fold for 0.99 contained only one class)
python proteomics_preprocessing.py clas_scop --scop_select=56 --split_ratio_train=0.98 --working_folder=datasets/clas_scop_fold/clas_scop_fold56 --pretrained_folder=datasets/lm/lm_sprot_uniref --foldername=SCOP167-fold

#########################
#BPE LMs
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_uniref_nofragments_bpe100 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=True --minproteinexistence=4 --bpe=True --bpe_vocab_size=100
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_uniref_nofragments_bpe2000 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --drop_fragments=True --minproteinexistence=4 --bpe=True --bpe_vocab_size=2000
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_uniref_noX_bpe100 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --bpe=True --bpe_vocab_size=100 --exclude_aas=["X"]
#python proteomics_preprocessing.py lm_sprot --working_folder=datasets/lm/lm_sprot_uniref_noX_bpe2000 --ignore_clusters=False --sampling_ratio=[.9,.05,.05] --sampling_method_train=1 --sampling_method_valtest=3 --bpe=True --bpe_vocab_size=2000 --exclude_aas=["X"]

#BPE EC uniprots
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_bpe100 --pretrained_folder=datasets/lm/lm_sprot_uniref_nofragments_bpe100 --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --bpe=True --bpe_vocab_size=100
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_bpe2000 --pretrained_folder=datasets/lm/lm_sprot_uniref_nofragments_bpe2000 --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --bpe=True --bpe_vocab_size=2000
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_noX_bpe100 --pretrained_folder=datasets/lm/lm_sprot_uniref_noX_bpe100 --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --bpe=True --bpe_vocab_size=100 --exclude_aas=["X"]
#python proteomics_preprocessing.py clas_ec --drop_ec7=True --working_folder=datasets/clas_ec/clas_ec_ec50_level1_noX_bpe2000 --pretrained_folder=datasets/lm/lm_sprot_uniref_noX_bpe2000 --level=1 --include_NoEC=False --dataset="uniprot" --sampling_method_train=1 --sampling_method_valtest=3 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1] --save_prev_ids=True --bpe=True --bpe_vocab_size=2000 --exclude_aas=["X"]

##################################
# deepre via accessions
#python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_deepre_accessions_level0 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="deepre_accessions" --sampling_method_train=1 --sampling_method_valtest=1 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1]
#python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_deepre_accessions_level1 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=False --dataset="deepre_accessions" --sampling_method_train=1 --sampling_method_valtest=1 --ignore_pretrained_clusters=True --sampling_ratio=[.8,.1,.1]

#5-fold CV
python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_deepre_accessions_level0_5foldcv --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="deepre_accessions" --nfolds=5 --sampling_method_train=1 --sampling_method_valtest=1 --ignore_pretrained_clusters=True
python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_deepre_accessions_level1_5foldcv --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=False --dataset="deepre_accessions" --nfolds=5 --sampling_method_train=1 --sampling_method_valtest=1 --ignore_pretrained_clusters=True

##################################
# ecpred via accessions
# binary label (all concatenated)
for x in {1..6}
do
    python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_ecpred_accessions_level0_EC$x --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="ecpred_accessions$x"
done
#6+1 class data
for x in {1..6}
do
    python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_ecpred_accessions_level1_EC$x --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=True --dataset="ecpred_accessions$x"
done
#6+1 class data
#for x in {1..6}
#do
#    python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_ecpred_accessions_level1_EC$xbinary --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=True --dataset="ecpred_accessions$xX"
#done

#python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_ecpred_accessions_level1_redundant_EC1 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=1 --include_NoEC=True --dataset="ecpred_accessions1" --ecpred_accessions_add_redundancy=True --ecpred_accessions_train_val_ratio=1.0
#python proteomics_preprocessing.py clas_ec --working_folder=datasets/clas_ec/clas_ec_ecpred_accessions_level0_redundant_EC1 --pretrained_folder=datasets/lm/lm_sprot_uniref --level=0 --include_NoEC=True --dataset="ecpred_accessions1" --ecpred_accessions_add_redundancy=True --ecpred_accessions_train_val_ratio=1.0

##################################
# go (using deeprotein data)
python proteomics_preprocessing.py clas_go_deeprotein --train_on_cafa3_original=False --eval_on_cafa3_test=True --working_folder=datasets/clas_go/clas_go_deeprotein_sp_train_cafa_test --pretrained_folder=datasets/lm/lm_sprot_uniref
#python proteomics_preprocessing.py clas_go_deeprotein --train_on_cafa3_original=False --eval_on_cafa3_test=False --working_folder=datasets/clas_go/clas_go_deeprotein_sp_train_deepgo_test --pretrained_folder=datasets/lm/lm_sprot_uniref
#python proteomics_preprocessing.py clas_go_deeprotein --train_on_cafa3_original=True --eval_on_cafa3_test=True --working_folder=datasets/clas_go/clas_go_deeprotein_cafa_train_cafa_test --pretrained_folder=datasets/lm/lm_sprot_uniref
#python proteomics_preprocessing.py clas_go_deeprotein --train_on_cafa3_original=True --eval_on_cafa3_test=False --working_folder=datasets/clas_go/clas_go_deeprotein_cafa_train_deepgo_test --pretrained_folder=datasets/lm/lm_sprot_uniref


