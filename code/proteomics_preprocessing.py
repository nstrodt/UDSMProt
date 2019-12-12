'''Preparing datasets for language modelling, classification and sequence annotation
'''

import fire
# This needs to be called in order to load local implemented submodules
import os
import sys
#module_path = os.path.abspath(os.path.join('../'))
#if module_path not in sys.path:
#    sys.path.append(module_path)

from tqdm import tqdm as tqdm
from utils.proteomics_utils import *
from utils.dataset_utils import *

#for log header
import datetime
import subprocess

import itertools
import ast
from pandas import concat

#for kinase dicts
from collections import defaultdict
######################################################################################################
#DATA PATHS
######################################################################################################
#path to the data directory
data_path = Path('../data')

#SEQUENCE DATA
#ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz (this is swissprot not the whole uniprot)
path_sprot=data_path/'uniprot_sprot_2017_03.xml'

#CLUSTER DATA (default path)
#ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz unzipped 100GB file (whole uniprot not just swissprot)
path_uniref = data_path/"uniref50_2017_03.xml"
#path to the data directory also uploaded to git
git_data_path = Path('../git_data')
# Enzyme Classifaction Datasets
path_ec_knna = git_data_path/"suppa.txt"
path_ec_knnb = git_data_path/"suppb.txt"

#deepre and ecpred datasets as prepared via utils/Parse xxx Data.ipynb
path_ec_deepre = git_data_path/"DEEPre_data.pkl"
path_ec_ecpred = git_data_path/"ECPred_data.pkl"
#path to the temporary directory for dataframes
tmp_data_path = Path("../tmp_data")#should be moved to ./data_tmp or similar
tmp_data_path.mkdir(exist_ok=True)

######################################################################################################
#AUX FUNCTIONS
######################################################################################################
def load_uniprot(source=path_sprot,parse_features=[]):
    '''parse and load uniprot xml
    parse_features: list of features to be parsed (modified residue for PTM etc.)
    '''
    pf="_"+"_".join([p.replace(" ","_") for p in parse_features])
    path_pkl = tmp_data_path/(source.stem+(pf if len(pf)>1 else "")+".pkl")
    if path_pkl.exists():
        print("Loading uniprot pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl, "not found. Parsing uniprot xml...")
        df=parse_uniprot_xml(source,parse_features=parse_features)
        df.to_pickle(path_pkl)
        return df

def load_uniref(source=path_uniref,source_uniprot=path_sprot):
    '''parse and load uniref xml
    '''
    path_pkl = tmp_data_path/(source.stem+("_"+source_uniprot.stem if source_uniprot is not None else "full")+".pkl")
    if path_pkl.exists():
        print("Loading uniref pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl,"not found. Parsing uniref...")
        df_selection = None if source_uniprot is None else load_uniprot(source_uniprot)
        df = parse_uniref(source,max_entries=0,parse_sequence=False,df_selection=df_selection)
        df.to_pickle(path_pkl)
        return df

def load_cdhit(df, cluster_type, dataset):
    '''loads/creates cluster dataframe for a given sequence dataframe (saves pickled result for later runs)
    cluster_type: cdhit05 (uniref-like cdhit down to threshold 0.5 in three stages), cdhit04 (direct cdhit down to threshold 0.4) and recalculate_cdhit05 and recalculate_cdhit04 (which do not make use of cached clusters)
    dataset: e.g. sprot determines the name of the pkl file
    '''
    path_pkl = tmp_data_path/(cluster_type+"_"+dataset+".pkl")
    
    if path_pkl.exists() and not(cluster_type == "recalculate_cdhit04" or cluster_type == "recalculate_cdhit05"):
        print("Loading cdhit pkl from",path_pkl)
        return pd.read_pickle(path_pkl)
    else:
        print(path_pkl,"not found. Running cdhit...")
        if(cluster_type=="cdhit05" or cluster_type=="recalculate_cdhit05"):#uniref-like cdhit 0.5 in three stages
            threshold=[1.0,0.9,0.5]
            alignment_coverage=[0.0,0.9,0.8]
        else:#direct cdhit clustering to 0.4
            threshold=[0.4]
            alignment_coverage=[0.0]
        df=clusters_df_from_sequence_df(df[["sequence"]],threshold=threshold,alignment_coverage=alignment_coverage)
        df.to_pickle(path_pkl)
        return df
        
def write_log_header(path, kwargs, filename="logfile.log",append=True):
    path.mkdir(exist_ok=True)
    (path/"models").mkdir(exist_ok=True)

    if "self" in kwargs:
        del kwargs["self"]
    
    print("======================================\nCommand:"," ".join(sys.argv))
    time = datetime.datetime.now() 
    print("started at ",time)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    print("Commit:",commit)
    
    
    print("\nArguments:")
    for k in sorted(kwargs.keys()):
        print(k,":",kwargs[k])
    print("")

    filepath=path/filename
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        f.write("\n\nCommand "+" ".join(sys.argv)+"\n")
        f.write("started at "+str(time)+"\n")
        f.write("Commit "+str(commit)+"\n")
        f.write("\nArguments:\n")
        for k in sorted(kwargs.keys()):
            f.write(k+": "+str(kwargs[k])+"\n")
        f.write("\n") 

def write_log(path, text, filename="logfile.log",append=True):
    filepath=path/filename
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        f.write(str(text)+"\n")
######################################################################################################
class Preprocess(object):
    """Preprocessing class (implemented as class to allow different function calls from CLI)."""

    #############################
    # GENERAL
    #############################    
    def _preprocess_default(self,path,df,df_cluster,pad_idx=0,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,drop_fragments=False,minproteinexistence=0,exclude_aas=[],nfolds=None,sampling_ratio=[0.8,0.1,0.1],ignore_pretrained_clusters=False,save_prev_ids=True,regression=False,sequence_output=False,bpe=False,bpe_vocab_size=100, sampling_method_train=1, sampling_method_valtest=3,subsampling_ratio_train=1.0,randomize=False,random_seed=42,mask_idx=1,tok_itos_in=[],label_itos_in=[],pretrained_path=None, df_redundancy=None, df_cluster_redundancy=None):
        '''internal routine for lm preprocessing
        path: Pathlib model path
        df_cluster: clusters dataframe
        pad_idx: index of the padding token

        sequence_len_min_aas: only consider sequences of at least sequence_len_min_aas aas length 0 for no restriction
        sequence_len_max_aas: only consider sequences up to sequence_len_max_aas aas length 0 for no restriction
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization) 0 for no restriction
        drop_fragments: drop proteins with fragment marker (requires fragment column in df)
        minproteinexistence: drop proteins with proteinexistence < value (requires proteinexistence column in df)
        exclude_aas: drop sequences that contain aas in the exclude list e.g. exclude_aas=["B","J","X","Z"] for sequences with only canonical AAs ( B = D or N, J = I or L, X = unknown, Z = E or Q)

        nfolds: if None perform a single split according to sampling_ratio else nfold CV split
        sampling_ratio: sampling ratios for train/val/test
        regression: regression task
        sequence_output: output/label is a sequence
        ignore_pretrained_clusters: do not take into account existing clusters from the previous LM step during train-test-split

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        subsampling_ratio_train: artificially reduce train (clusters) to specified ratio (1.0 for full dataset)
        mask_idx: index of the mask token (for BERT masked LM training) None for none

        tok_itos_in: allows to pass tok_itos_in (used for trembl processing) will use pretrained tok_itos if an empty list is passed

        df_redundancy: optional df with same structure as df with additional redundant sequences
        df_cluster_redundancy: optional cluster df including redundant and original sequences
        '''

        #filter by fragments
        if(drop_fragments):
            if("fragment" in df.columns):
                df = df[df.fragment == 0]
                if(df_redundancy is not None):
                    df_redundancy = df_redundancy[df_redundancy.fragment == 0]
            else:
                print("Warning: could not drop fragments as fragment column is not available")
        #filter by proteinexistence
        if(minproteinexistence>0):
            if("proteinexistence" in df.columns):
                df = df[df.proteinexistence >= minproteinexistence]
                if(df_redundancy is not None):
                    df_redundancy = df_redundancy[df_redundancy.proteinexistence >= minproteinexistence]
            else:
                print("Warning: could not filter by protein existence as proteinexistence column is not available")
        #filter by non-canonical AAs
        if(len(exclude_aas)>0):
            df = filter_aas(df, exclude_aas)
            if(df_redundancy is not None):
                df_redundancy = filter_aas(df_redundancy, exclude_aas)
        #filter by aa length
        if(sequence_len_min_aas>0):
            df = df[df.sequence.apply(len)>sequence_len_min_aas]
            if(df_redundancy is not None):
                df_redundancy = df_redundancy[df_redundancy.sequence.apply(len)>sequence_len_min_aas]
        if(sequence_len_max_aas>0):
            df = df[df.sequence.apply(len)<sequence_len_max_aas]
            if(df_redundancy is not None):
                df_redundancy = df_redundancy[df_redundancy.sequence.apply(len)<sequence_len_max_aas]

        if(bpe):
            spt=sentencepiece_tokenizer(path if pretrained_path is None else pretrained_path,df,vocab_size=bpe_vocab_size)

        if(len(tok_itos_in)==0 and pretrained_path is not None):            
            tok_itos_in = np.load(pretrained_path/"tok_itos.npy")
        if(pretrained_path is not None and ignore_pretrained_clusters is False):
            if(nfolds is None):
                train_ids_prev = np.load(pretrained_path/"train_IDs_prev.npy")
                val_ids_prev = np.load(pretrained_path/"val_IDs_prev.npy")
                test_ids_prev = np.load(pretrained_path/"test_IDs_prev.npy")
            else:
                cluster_ids_prev = np.load(pretrained_path/"cluster_IDs_CV_prev.npy")
        else:
            if(nfolds is None):
                train_ids_prev = []
                val_ids_prev = []
                test_ids_prev =[]
            else:
                cluster_ids_prev = []

        prepare_dataset(path,df,tokenizer=(spt.tokenize if bpe is True else list_tokenizer),pad_idx=pad_idx,sequence_len_max_tokens=sequence_len_max_tokens,mask_idx=mask_idx,tok_itos_in=tok_itos_in,label_itos_in=label_itos_in,df_seq_redundant=df_redundancy,regression=regression,sequence_output=sequence_output)
        ids_current_all = np.load(path/"ID.npy") #filtered by sequence length
        ids_current = np.intersect1d(list(df.index),ids_current_all) #ids_current_all might contain more sequences
        ids_current_redundancy = [] if df_cluster_redundancy is None else  np.intersect1d(list(df_redundancy.index),ids_current_all)
        if(nfolds is None):
            train_test_split(path,ids_current,ids_current_all,df_cluster,train_ids_prev=train_ids_prev,val_ids_prev=val_ids_prev,test_ids_prev=test_ids_prev,sampling_ratio=sampling_ratio,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,subsampling_ratio_train=subsampling_ratio_train,randomize=randomize,random_seed=random_seed,save_prev_ids=save_prev_ids,ids_current_redundancy=ids_current_redundancy,df_cluster_redundancy=df_cluster_redundancy)
        else:
            cv_split(path,ids_current,ids_current_all,df_cluster,clusters_prev=cluster_ids_prev,nfolds=nfolds, sampling_method_train=sampling_method_train, sampling_method_valtest=sampling_method_valtest, randomize=randomize, random_seed=random_seed, save_prev_ids=save_prev_ids)

    #############################
    # LM
    ############################# 
    def lm_sprot(self, source_sprot=path_sprot,source_uniref=path_uniref,only_human_proteome=False,drop_fragments=False,minproteinexistence=0,exclude_aas=[],working_folder="./lm_sprot",pretrained_folder="",pad_idx=0,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,nfolds=None,sampling_ratio=[0.9,0.05,0.05],cluster_type="uniref",ignore_clusters=False,bpe=False,bpe_vocab_size=100,sampling_method_train=1, sampling_method_valtest=3,subsampling_ratio_train=1.0,randomize=False,random_seed=42,mask_idx=1):

        '''prepare sprot LM data
        
        only_human_proteome: filter only human proteome
        drop_fragments: drop proteins marked as fragments
        minproteinexistence: drop proteins with protein existence smaller than minproteinexistence
        working_folder: path of the data folder to be created (as string)
        pretrained_folder: path to pretrained folder (as string; empty string for none)
        pad_idx: index of the padding token
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization) 0 for no restriction
        
        nfolds: number of CV splits; None for single split
        sampling_ratio: sampling ratios for train/val/test
        ignore_clusters: do not use cluster information for train/test split
        cluster_type: source of clustering information (uniref, cdhit05: cdhit threshold 0.5 similar to uniref procedure, cdhit04: cdhit with threshold 0.4)

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        subsampling_ratio_train: portion of train clusters used
        pick_representative_for_val_test: just select a single representative per cluster for validation and test set
        mask_idx: index of the mask token (for BERT masked LM training) None for none
        '''
        print("Preparing sprot LM")

        LM_PATH=Path(working_folder)
        write_log_header(LM_PATH,locals())

        source_sprot = Path(source_sprot)

        df = load_uniprot(source=source_sprot)

        if(only_human_proteome):
            df = filter_human_proteome(df)
            print("Extracted {} human proteines".format(df.shape[0]))
        
        if(ignore_clusters is False):
            if(cluster_type[:6]=="uniref"):
                df_cluster = load_uniref(source_uniref,source_sprot)
            else:
                df_cluster = load_cdhit(df,cluster_type,source_sprot.stem)
        else:
            df_cluster = None
        self._preprocess_default(path=LM_PATH,df=df,df_cluster=df_cluster,pad_idx=pad_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,drop_fragments=drop_fragments,minproteinexistence=minproteinexistence,exclude_aas=exclude_aas,nfolds=nfolds,sampling_ratio=sampling_ratio,bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,subsampling_ratio_train=subsampling_ratio_train,randomize=randomize,random_seed=random_seed,mask_idx=mask_idx,pretrained_path=Path(pretrained_folder) if pretrained_folder !="" else None)
    #############################
    # CLASSIFICATION
    ############################# 
    def clas_scop(self,scop_select,foldername="SCOP167-superfamily",split_ratio_train=None,cdhit_threshold=0.5,cdhit_alignment_coverage=0.8,working_folder="./clas_scop",pretrained_folder="./lm_sprot",pad_idx=0,mask_idx=1,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0,bpe=False,bpe_vocab_size=100, save_prev_ids=False):
        print("Preparing scop classification dataset #"+str(scop_select)+" ...")
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())

        data_path_scop = data_path /foldername
        datasets = []
        for r in data_path_scop.glob("neg-test.*"):
            x=str(r.stem)
            datasets.append(x[9:])
        datasets= list(set(datasets))
        datasets.sort()

        assert scop_select<len(datasets),"invalid scop_select"
        print("dataset",datasets[scop_select],"(",scop_select+1,"/",len(datasets),")")
        
        df_clas = generate_homology_scop(filename_postfix=datasets[scop_select], data_dir=data_path_scop, split_ratio_train=split_ratio_train,cdhit_threshold=cdhit_threshold,cdhit_alignment_coverage=cdhit_alignment_coverage)

        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_clas,pad_idx=pad_idx,mask_idx=mask_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=-1,sampling_method_valtest=-1, save_prev_ids=False)
    
    def clas_ec(self, source_sprot=path_sprot, source_uniref=path_uniref, dataset="uniprot",level=1,include_NoEC=False,minproteinexistence_NoEC=4,include_superclasses=False,ecpred_accessions_add_redundancy=False,ecpred_accessions_train_val_ratio=0.95,drop_incomplete=True,drop_ec7=True,drop_fragments=True,minproteinexistence=0,exclude_aas=[],single_label=True,working_folder="./clas_ec_sprot",pretrained_folder="./lm_sprot",pad_idx=0,mask_idx=1,sequence_len_min_aas=50,sequence_len_max_aas=5000,sequence_len_max_tokens=0,nfolds=None,sampling_ratio=[0.8,0.1,0.1],ignore_pretrained_clusters=False, cluster_type="uniref", ignore_clusters=False,bpe=False,bpe_vocab_size=100,sampling_method_train=1,sampling_method_valtest=3,subsampling_ratio_train=1.0,randomize=False,random_seed=42, save_prev_ids=False):
        '''prepare EC classification data
        dataset: uniprot (default uniprot), knna, knnb, deeppre aka knn_new, deepre_accession (via accessions provided on their homepage), ecpred_accessionsX (via accessions provided in their github repository; for ec class X=1..6 using all data for training), ecpred_accesionsXb (corresponding dataset for training a binary classifier using the corresponding training set of the respective class only)
        level: predict up a certain level in the EC hierarchy (level=1: first digit)
        include_NoEC: include non-enzymes as a separate class
        minproteinexistence_NoEC: keep only non-enzymes with proteinexistence equal to or larger than minproteinexistence_NoEC
        single_label: discard all multi-label entries
        include_superclasses: include also all corresponding superclasses as labels
        
        ecpred_accessions_add_redundancy: add redundant sequences from sprot to enlarge the ecpred training dataset
        ecpred_accessions_train_val_ratio: split original expred training set into training and validation set according to this ratio

        drop_incomplete: drop incomplete EC entries with - after truncation
        drop_ec7: drop seventh EC class
        drop_fragments: drop all entries with fragment status

        pretrained_folder: path to the language model folder (as string)
        working_folder: path of the data folder to be created (as string) (default name will be inserted if empty string is passed)

        pad_idx: index of the padding token
        sequence_len_max_tokens: only consider sequences up to sequence_len_max_tokens tokens length (after tokenization)
        nfolds: number of CV splits; None for single split
        sampling_ratio: sampling ratios for train/val/test
        ignore_pretrained_clusters: do not take into account existing clusters from the previous LM step during train-test-split

        bpe: apply BPE
        bpe_vocab_size: vocabulary size (including original tokens) for BPE

        sampling_method: sampling method for train test split as defined in dataset_utils
        subsampling_ratio_train: as defined above
        '''
        print("Preparing EC classification dataset "+dataset+" ...")

        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())

        source_sprot = Path(source_sprot)
        # just in case the user forgot to set include_NoEC to True
        if level == 0:
            include_NoEC = True
        
        #no redundancy by default
        df_clas_redundancy = None
        df_cluster_redundancy = None
        df_redundancy = None
        
        if(dataset =="uniprot"):
            df_uniprot = load_uniprot(source=source_sprot)
            
            df = ecs_from_uniprot(df_uniprot, level=level, drop_fragments=drop_fragments, drop_incomplete=drop_incomplete, include_NoEC=include_NoEC, minproteinexistence_NoEC=minproteinexistence_NoEC)
            if(ignore_clusters is False):#this presently uses uniprot clusters (ec would be more efficient but we don't know when to recalculate)
                if(cluster_type[:6]=="uniref"):
                    df_cluster = load_uniref(source_uniref,source_sprot)
                else:
                    df_cluster = load_cdhit(df,cluster_type,source_sprot.stem)
        elif(dataset == "knna"):
            assert(level==1)
            df = ecs_from_knna(path_ec_knna)
            df_cluster = None #random split
            ignore_pretrained_clusters = True
            drop_fragments = False
        elif(dataset == "knnb"):
            assert(level<=2)
            df = ecs_from_knnb(path_ec_knnb)
            df_cluster = None #random split
            ignore_pretrained_clusters = True
            drop_fragments = False
        elif(dataset == "deepre" or dataset=="knn_new"):
            # check if already computed and dumped!
            if os.path.isfile(str(CLAS_PATH/"cdhit04_clustering.csv")) & os.path.isfile(str(CLAS_PATH/"cdhit04_remaining_data.csv")):
                print("load dumped clustering")
                df_cluster = pd.read_csv(CLAS_PATH/"cdhit04_clustering.csv")
                df_cluster = df_cluster.set_index('ID')
                df = pd.read_csv(CLAS_PATH/"cdhit04_remaining_data.csv")
                df = df.set_index('ID')
                df.ecs_truncated = df.ecs_truncated.apply(lambda x: ast.literal_eval(x))
            else:
                df_uniprot = load_uniprot_xml(source_uniref)
                #df_uniprot=parse_uniprot_xml(path_sprot,max_entries=1000)
                knn_new_threshold=0.4
                knn_new_alignment_coverage=.0
                df_reduced, df, df_cluster = ecs_from_knn_new(df_uniprot, level=level, threshold=knn_new_threshold, alignment_coverage=knn_new_alignment_coverage, min_length=sequence_len_min_aas, max_length=sequence_len_max_aas, drop_fragments=drop_fragments, drop_incomplete=drop_incomplete, include_NoEC=include_NoEC, single_label=single_label)
                # store df_cluster since its heavy to compute
                df_cluster.to_csv(CLAS_PATH/"cdhit04_clustering.csv")
                df.to_csv(CLAS_PATH/"cdhit04_remaining_data.csv")
        elif(dataset=="deepre_accessions"):
            df = pd.read_pickle(path_ec_deepre)
            df = df.set_index("accesion")
            df["ecs"]=df.ec.apply(lambda x: [] if type(x)!=str else [x])
            if(cluster_type == "cdhit04"):
                df_cluster = load_cdhit(df,cluster_type,source_sprot.stem)
                sampling_method_train = 3 #pick all samples to cover full deepre dataset
                sampling_method_valtest = 3
            else:
                df_cluster = None #random split on representatives
            drop_fragments= False
            ignore_pretrained_clusters = True
        elif(dataset[:17]=="ecpred_accessions"):
            assert(include_NoEC is True)
            df = prepare_ecpred_data(path_ec_ecpred, ec_class=int(dataset[17]), full_train= (len(dataset)==18), train_val_ratio=ecpred_accessions_train_val_ratio, random_seed=random_seed)
            df_cluster = df #use predefined split
            if(ecpred_accessions_add_redundancy is True):
                df_redundancy = ecs_from_uniprot(load_uniprot(source=source_sprot), level=level, drop_fragments=drop_fragments, drop_incomplete=drop_incomplete, include_NoEC=include_NoEC, minproteinexistence_NoEC=minproteinexistence_NoEC)
                df_cluster_redundancy = load_uniref(source_uniref,source_sprot)
                #train_ids = list(df[df.cluster_ID==0].index)
                #train_clusters = np.unique(df_cluster_redundancy[df_cluster_redundancy.index.isin(train_ids)].cluster_ID)
                #train_ids_extended = list(df_uniref.cluster_ID.isin(train_clusters).index)
                #print("train_ids",len(train_ids),"train_clusters",len(train_clusters),"train_ids_extended",len(train_ids_extended))
                #df_extended = df_sprot[df_sprot.index.isin(train_ids_extended)].copy()
                #df_extended["cluster_ID"]=0
                #print("original df",len(df), "train:", len(df[df.cluster_ID==0]), "extended",len(df_extended))
                
                
            sampling_method_train = -1
            sampling_method_valtest = -1
            ignore_pretrained_clusters = True
            drop_fragments = False

        df_ec, ec_itos = ecs_from_df(df,level=level,include_NoEC=include_NoEC,drop_incomplete=drop_incomplete,drop_ec7=drop_ec7,single_label=single_label,include_superclasses=include_superclasses)

        if(df_redundancy is not None):
            df_clas_redundancy, _ = ecs_from_df(df_redundancy,level=level,include_NoEC=include_NoEC,drop_incomplete=drop_incomplete,drop_ec7=drop_ec7,single_label=single_label,include_superclasses=include_superclasses)
               
        if(single_label is True and include_superclasses is False):
            df_ec["label"]=df_ec.ecs_truncated.apply(lambda x: x[0])
            df_clas = df_ec
            if df_redundancy is not None:
                df_clas_redundancy["label"]=df_clas_redundancy.ecs_truncated.apply(lambda x: x[0])
        else:
            df_ec["label"]=df_ec.ecs_truncated
            df_clas = df_ec
            if df_redundancy is not None:
                df_clas_redundancy["label"]=df_clas_redundancy.ecs_truncated

        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_cluster,label_itos_in=ec_itos,pad_idx=pad_idx,mask_idx=mask_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,drop_fragments=drop_fragments,minproteinexistence=minproteinexistence,exclude_aas=exclude_aas,nfolds=nfolds,sampling_ratio=sampling_ratio, ignore_pretrained_clusters=ignore_pretrained_clusters,bpe=bpe,bpe_vocab_size=bpe_vocab_size,sampling_method_train=sampling_method_train,sampling_method_valtest=sampling_method_valtest,subsampling_ratio_train=subsampling_ratio_train,randomize=randomize,random_seed=random_seed, save_prev_ids=save_prev_ids,df_redundancy=df_clas_redundancy,df_cluster_redundancy=df_cluster_redundancy)
    def clas_go_deepgoplus(self,cafa_data=True,nmin_train=50,ont_filter=None,working_folder="./clas_go_deepgoplus",pretrained_folder="./lm_sprot",pad_idx=0,mask_idx=1,sequence_len_min_aas=0,sequence_len_max_aas=0,sequence_len_max_tokens=0):
        '''prepares GO classification data based on Deepgoplus data http://deepgoplus.bio2vec.net/data/
        requires data_cafa.tar.gz extracted in data or data_2016.tar.gz extracted in data as deepgoplus_data_cafa or deepgoplus_data_2016
        '''
        print("Preparing deepgoplus GO classification dataset...")
        if(cafa_data is True):
            source_folder="../data/deepgoplus_data_cafa"
            use_valid=True
        else:
            source_folder="../data/deepgoplus_data_2016"
            use_valid=False
        
        CLAS_PATH = Path(working_folder)
        LM_PATH=Path(pretrained_folder) if pretrained_folder!="" else None
        write_log_header(CLAS_PATH,locals())
        
        source_path = Path(source_folder)
        
        df_clas = prepare_deepgoplus_data(source_path/"train_data_train.pkl" if use_valid else source_path/"train_data.pkl",source_path/"train_data_valid.pkl" if use_valid else None,source_path/"test_data.pkl",str(source_path/"go.obo"),nmin_train=nmin_train,propagate_scores=False,ont_filter=ont_filter)
        
        df_cluster = df_clas #predetermined split

        self._preprocess_default(path=CLAS_PATH,pretrained_path=LM_PATH,df=df_clas,df_cluster=df_cluster,pad_idx=pad_idx,mask_idx=mask_idx,sequence_len_min_aas=sequence_len_min_aas,sequence_len_max_aas=sequence_len_max_aas,sequence_len_max_tokens=sequence_len_max_tokens,sampling_method_train=-1,sampling_method_valtest=-1,ignore_pretrained_clusters=True)
        
    #############################
    # GENERATE PSSM FEATURES 
    #############################
    def features_pssm(self, query_folder, db_folder, evalue=.001, num_iterations=3, sigmoid=True, train_db=False, n_processes=8, blast_dbs_path=Path("../blast_dbs")):
        compute_pssm_features(query_folder, db_folder, evalue, num_iterations, sigmoid, train_db, n_processes, blast_dbs_path)
if __name__ == '__main__':
    fire.Fire(Preprocess)
