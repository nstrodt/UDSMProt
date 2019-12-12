import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split as tts

#console
from tqdm import tqdm as tqdm

import sentencepiece as sp 
############################################################################
# tokenizers
############################################################################
def list_tokenizer(seq): 
    '''default tokenizer just returns a list'''
    return [t for t in seq]

class sentencepiece_tokenizer:
    '''a sentencepiece tokenizer'''
    def __init__(self,path,df_sequences=None,vocab_size=100,model_type='bpe',add_dummy_prefix=False):
        '''constructor: loads model if it exists or trains a new one
        path: pathlib path to working folder
        df_sequences: dataframe with column sequences
        vocab_size: total size of the vocabulary (including original tokens)
        model_type: sentencepiece model_type
        add_dummy_prefix: sentencepiece add_dummy_prefix
        '''
        file_model = path/(model_type+"_"+str(vocab_size)+".model")
        if(file_model.exists()):
            print("Loading existing sentencepiece tokenizer...")
            self.load(file_model)
        else:
            print("Training sentencepiece tokenizer from scratch...")
            self.train(path,df_sequences,vocab_size,model_type,add_dummy_prefix)
            self.load(file_model)


    def load(self,path):
        '''internal routine: loads model'''
        self.model = sp.SentencePieceProcessor()
        PATH=Path(path)
        self.model.Load(str(PATH))
        
    def train(self,path,df_sequences,vocab_size=100,model_type='bpe',add_dummy_prefix=False):
        '''internal routine: trains model'''
        #save sequences
        tmp_path = Path('./sentencepiece.txt')
        with tmp_path.open("w", encoding ="utf-8") as f:
            for i,s in enumerate(df_sequences.sequence):
                if(i==0):
                    f.write(s)
                else:
                    f.write("\n"+s)
        #train
        sp.SentencePieceTrainer.Train('--input=./sentencepiece.txt  --model_prefix=sentencepiece --vocab_size='+str(vocab_size)+' --model_type='+model_type+' --add_dummy_prefix='+str(add_dummy_prefix))
        tmp_path.unlink()
        #move output
        src_model=Path("sentencepiece.model")
        dest_model=path/(model_type+"_"+str(vocab_size)+".model")
        src_vocab=Path("sentencepiece.vocab")
        dest_vocab=path/(model_type+"_"+str(vocab_size)+".vocab")
        src_model.rename(dest_model)
        src_vocab.rename(dest_vocab)

    
    def tokenize(self,seq):
        '''tokenizes a given sequence using the internal model'''
        tok_bpe = self.model.EncodeAsPieces(seq)
        try: 
            tok_bpe = [t.decode('UTF-8') for t in tok_bpe] 
        except AttributeError: pass
        return tok_bpe

#text=["AAEFDD","DDDDDFFF","FFKKJKJUIUE","FJKJKJKEY","AAEGGFF","FFFKKK"]
#df_text = pd.DataFrame(text, columns=['sequence'])
#spt = sentencepiece_tokenizer(pathlib.Path("./"),df_text,vocab_size=20)
#for t in text:
#    print(spt.tokenize(t))

##########################################################################
#prepare dataset from dataframe
##########################################################################
def prepare_dataset(path,df_seq,tokenizer=list_tokenizer,sequence_len_min_tokens=0, sequence_len_max_tokens=0, tok_itos_in=[],pad_idx=0, label_itos_in=[], insert_bos=True, insert_eos=False, sequence_output=False, insert_oov=False,max_entries=0,regression=False,max_vocab=60000,min_freq=2, mask_idx=None, df_seq_redundant=None):
    '''
    Creates set of numerical arrays from a dataframe for further processing.

    Parameters:
    path: output path as string
    df_seq: pandas dataframe with data; columns ID (as index), sequence (and optionally label)
    tokenizer: reference to tokenizer function
    sequence_len_min_tokens: minimum length of tokenized sequence
    sequence_len_max_tokens: maximum length of tokenized sequence (0 for no length restriction)
    tok_itos_in: optional int to string mapping for tokens (to obtain a consistent tokenization with previous pretraining tasks)
    label_itos_in: optional int to string mapping for labels (in case the mapping was already applied in the dataframe) special tokens are _bg_ for background and _none_ for ignoring
    insert_bos: insert bos token (_bos_)
    insert_eos: insert eos token (_eos_)
    pad_idx: ID of the padding token
    insert_oov: insert oov token into vocabulary (otherwise oov will be mapped to padding token)
    
    sequence output: output is a sequence (will add padding token to target labels)
    
    max_entries: return only the first max_entries entries (0 for all)
    regression: labels are continuous for regression task

    max_vocab: only include the most frequent max_vocab tokens in the vocabulary
    min_freq: only include tokens in the vocabulary that have more than min_freq occurrences

    mask_idx: id of mask token (for BERT pretraining) None for none
    
    df_seq_redundant: optional dataframe with redundant sequences
    
    creates in the directory path:
        tok.npy input sequence mapped to integers
        label.npy labels mapped to integers
        tok_itos.npy integer to token mapping
        label_itos.npy integer to label mapping
    
    TODO parallelize tokenization
    TODO take care of multilabel-annotation
    '''
    label_none= "_none_" #special label: none (i.e. irrelevant) token for annotation labels e.g. for padding/eos but also for irrelevant phosphorylation site predictions
    label_bg = "_bg_" #special label: background token for annotation labels

    token_oov="_oov_"
    token_pad="_pad_"
    token_bos="_bos_"
    token_eos="_eos_"
    token_mask="_mask_"

    assert(len(np.unique(df_seq.index))==len(df_seq)),"Error in prepare_dataset: df_seq index contains duplicates."
    assert(df_seq_redundant is None or len(np.unique(df_seq_redundant.index))==len(df_seq_redundant)),"Error in prepare_dataset: df_seq_redundant index contains duplicates."
    
    print("\n\nPreparing dataset:", len(df_seq), "rows in the original dataframe.")
    #create target path
    PATH = Path(path)
    PATH.mkdir(exist_ok=True)

    #delete aux. files if they exist
    aux_files = ['val_IDs_CV.npy','val_IDs.npy','train_IDs.npy', 'train_IDs_CV.npy', 'test_IDs.npy','test_IDs_CV.npy']
    for f in aux_files:
        p=PATH/f
        if(p.exists()):
            p.unlink()

    if(df_seq_redundant is None):
        df = df_seq
    else:
        ids_extra = np.setdiff1d(list(df_seq_redundant.index),list(df_seq.index))#in doubt take entries from the original df_seq
        common_columns = list(np.intersect1d(df_seq.columns,df_seq_redundant.columns))
        df = pd.concat([df_seq[common_columns],df_seq_redundant[common_columns].loc[df_seq_redundant.index.intersection(ids_extra)]]) 
        #print("df_seq.columns",df_seq[common_columns].columns,"df_seq_redundant.columns",df_seq_redundant[common_columns].columns,"df.columns",df.columns)
        print("label_itos_in",label_itos_in)
    #label-encode label column
    if "label" in df.columns: #label is specified
        if(regression):
            df["label_enc"]=df.label
        else:#categorical label
            if(len(label_itos_in)>0): #use the given label_itos
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list
                    df["label_enc"]=df.label
                else:#label is a single entry
                    df["label_enc"]=df.label.astype('int64')
                label_itos=label_itos_in
            else: #create label_itos
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list
                    label_itos = np.sort(np.unique([x for s in list(df.label) for x in s]))
                else:#label is a single entry
                    label_itos=list(np.sort(df.label.unique()).astype(str))
                
                numerical_label=False
                if(not(isinstance(label_itos[0],str))):
                    numerical_label=True
                    label_itos = [str(x) for x in label_itos]
            
            if(sequence_output):#annotation dataset: make sure special tokens are available (and make sure that the label_none item is at pad_idx- otherwise padding won't work as intended)
                label_itos_new = label_itos.copy()
                if(label_itos_new[pad_idx]!=label_none):
                    if(label_none in label_itos_new):
                        label_itos_new.remove(label_none)
                if(not(label_none in label_itos_new)):
                    label_itos_new.insert(pad_idx,label_none)
                if(not(label_bg in label_itos)):
                    label_itos_new.insert(pad_idx+1,label_bg)
                if(len(label_itos_in)>0):#apply new mapping to existing mapped labels
                    label_itoi_transition={idx:label_itos_new.index(label_itos[idx]) for idx in range(len(label_itos))}
                    #label is a list: multilabel classification
                    df["label_enc"]=df.label_enc.apply(lambda x:[label_itoi_transition[y] for y in x])   
                label_itos = label_itos_new
                
            np.save(PATH/"label_itos.npy",label_itos)

            if(len(label_itos_in)==0):#apply mapping to integer
                label_stoi={s:i for i,s in enumerate(label_itos)}
                if(isinstance(df.label.iloc[0],list) or isinstance(df.label.iloc[0],np.ndarray)):#label is a list: multilabel classification
                    df["label_enc"]=df.label.apply(lambda x:[label_stoi[str(y)] for y in x])
                else:#single-label classification
                    df["label_enc"]=df.label.apply(lambda x:label_stoi[str(x)]).astype('int64')

            #one-hot encoding for multilabel classification
            if(sequence_output is False and (isinstance(df.label_enc.iloc[0],list) or isinstance(df.label_enc.iloc[0],np.ndarray))):#multi-label classification
                #one-hot encoding
                def one_hot_encode(x,classes):
                    y= np.zeros(classes)#,dtype=int) #float expected in multilabel_soft_margin_loss
                    for i in x:
                        y[i]=1
                    return y
                df["label_enc"]=df.label_enc.apply(lambda x: one_hot_encode(x,len(label_itos)))

    if(sequence_output):
        label_stoi={s:i for i,s in enumerate(label_itos)}           
            
    #tokenize text (to be parallelized)
    tok = []
    label = []
    ID = []
    for index, row in tqdm(df.iterrows()):
        item_tok = tokenizer(row['sequence'])
        if(insert_bos):
            item_tok=[token_bos]+item_tok
        if(insert_eos):
            item_tok=item_tok +[token_eos]
        if(sequence_len_min_tokens>0 and len(item_tok)<sequence_len_min_tokens):
            continue
        if(sequence_len_max_tokens>0 and len(item_tok)>=sequence_len_max_tokens):
            continue
        tok.append(item_tok)
        if("label" in df.columns):
            if(sequence_output is False):
                label.append(row["label_enc"])
            else:
                label_tmp=list(row["label_enc"])
                if(insert_bos):
                    label_tmp = [label_stoi[label_none]] + label_tmp
                if(insert_eos):
                    label_tmp = label_tmp + [label_stoi[label_none]]
                label.append(label_tmp)
        ID.append(index)
        if(max_entries>0 and len(tok)==max_entries):
            break
        
    #turn into integers
    if(len(tok_itos_in)==0): #create itos mapping
        freq = Counter(p for o in tok for p in o)
        tok_itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
        if(insert_oov is True):
            tok_itos.append(token_oov)
        if(mask_idx is None):
            tok_itos.insert(pad_idx,token_pad)
        else:#order matters
            if(pad_idx<mask_idx):
                tok_itos.insert(pad_idx,token_pad)
                tok_itos.insert(mask_idx,token_mask)
            else:
                tok_itos.insert(mask_idx,token_mask)
                tok_itos.insert(pad_idx,token_pad)
    else:#use predefined itos mapping
        tok_itos = tok_itos_in
    
    np.save(PATH/"tok_itos.npy",tok_itos)
    print("tok_itos (", len(tok_itos), "items):",list(tok_itos))
    if("label" in df.columns and not(regression)):
        print("label_itos (", len(label_itos), "items):",list(label_itos) if len(label_itos)<20 else list(label_itos)[:20],"" if len(label_itos)<20 else "... (showing first 20 items)")
    
    tok_stoi = defaultdict(lambda:(len(tok_itos) if insert_oov else pad_idx), {v:k for k,v in enumerate(tok_itos)})
    tok_num = np.array([[tok_stoi[o] for o in p] for p in tok])
    np.save(PATH/"tok.npy",tok_num)
    print("Saved",len(tok),"rows (filtered based on sequence length).")
    
    np.save(PATH/"ID.npy",ID)
    if("label" in df.columns):
        if not(regression):
            np.save(PATH/"label.npy",np.array(label).astype(np.int64))
        else:
            np.save(PATH/"label.npy",np.array(label))

##########################################################################
#train-test split
##########################################################################
# local definitions
def pick_random_representatives_from_clusters(cluster_ids, df_cluster):
    df = (df_cluster[df_cluster.cluster_ID.isin(cluster_ids)]).copy()
    if(len(df)==0):
        return []
    df["ID"] = df.index
    series_list = df.groupby("cluster_ID").ID.agg(lambda x: list(x))
    series_selected = series_list.apply(lambda x: x[np.random.choice(range(0,len(x)))])
    return list(series_selected.values)

def pick_representative_from_clusters(cluster_ids, df_cluster, pick_first_member_instead= True):
    representatives = list((df_cluster[df_cluster.cluster_ID.isin(cluster_ids) & df_cluster.representative == True]).index)
    if(pick_first_member_instead):
        #check for cluster we haven't picked
        available_clusters = list(df_cluster.loc[representatives].cluster_ID.unique())
        missing_clusters = np.setdiff1d(cluster_ids,available_clusters)
        if(len(missing_clusters)>0):
            print("INFO: For",len(missing_clusters)," clusters no representative could be found (e.g. due to isoforms or filtering rules)- picking respective first cluster members instead.")
            df = (df_cluster[df_cluster.cluster_ID.isin(missing_clusters)]).copy()
            df["ID"] = df.index
            representatives = representatives + list(df.groupby("cluster_ID").ID.agg(lambda x: list(x)[0]).values)
    return representatives

def find_clusters_from_members(ids_subset, df_cluster):
    # loc with missing values https://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-deprecate-loc-reindex-listlike
    return list(df_cluster.loc[df_cluster.index.intersection(ids_subset)]["cluster_ID"].unique())

def pick_all_members_from_clusters(cluster_ids, df_cluster):
    return list(df_cluster[df_cluster.cluster_ID.isin(cluster_ids)].index)

def print_set_summary(train_ids_current,val_ids_current,test_ids_current,train_clusters,val_clusters,test_clusters):
    total_ids_current=len(train_ids_current)+len(val_ids_current)+len(test_ids_current)
    print("samples:",len(train_ids_current),len(val_ids_current),len(test_ids_current),"total:",total_ids_current, "ratio:",len(train_ids_current)/total_ids_current,len(val_ids_current)/total_ids_current,len(test_ids_current)/total_ids_current)
    total_clusters_current=len(train_clusters)+len(val_clusters)+len(test_clusters)
    print("clusters:",len(train_clusters),len(val_clusters),len(test_clusters),"total:",total_clusters_current, "ratio:",len(train_clusters)/total_clusters_current,len(val_clusters)/total_clusters_current,len(test_clusters)/total_clusters_current)

def split_clusters_nfolds(df_cluster, nfolds=10, randomize=False, random_seed=42):
    '''nfold splitting of a cluster df and returns list of cluster IDs'''
    np.random.seed(random_seed)

    #cids=list(df_cluster.groupby('cluster_ID').size().sort_values(ascending=False).cluster_ID)
    cids, cid_counts=np.unique(df_cluster.cluster_ID, return_counts=True)
    if(randomize):
        cids = np.random.permutation(cids)
    else:
        ids_sorted = list(np.argsort(cid_counts, kind="stable")[::-1])
        cids=cids[ids_sorted]
    splits = [[] for _ in range(nfolds)]
    for i,c in enumerate(cids):
        splits[i%nfolds].append(c)
    return splits

def split_clusters_single(df_cluster, ratios, randomize=False, random_seed=42):
    '''splits a cluster df into parts (ratios as specified in ratios) and returns list of cluster IDs'''
    nfolds = len(ratios)
    np.random.seed(random_seed)
    #cids=list(df_cluster.groupby('cluster_ID').size().sort_values(ascending=False).cluster_ID)
    cids, cid_counts=np.unique(df_cluster.cluster_ID, return_counts=True)
    if(randomize):
        cids = np.random.permutation(cids)
    else:
        ids_sorted = list(np.argsort(cid_counts, kind="stable")[::-1])
        cids=cids[ids_sorted]
    splits = [[] for _ in range(nfolds)]
    i=-1
    for c in cids:
        i=(i+1)%nfolds
        for j in range(i,i+len(ratios)):
            if(len(splits[j%nfolds])/len(df_cluster)<ratios[j%nfolds]):
                i = j%nfolds
                splits[i].append(c)
                break
    return splits



def train_test_split(path, ids_current, ids_current_all, df_cluster, train_ids_prev=[], val_ids_prev=[], test_ids_prev=[], sampling_ratio=[0.8,0.2,0.0], sampling_method_train=1, sampling_method_valtest=3, subsampling_ratio_train=1.0, randomize=False, random_seed=42, save_prev_ids=False, ids_current_redundancy=None, df_cluster_redundancy=None):
    '''
    performs train-test split using the output of prepare_dataset

    Parameters:
    path: working directory
    ids_current: list of IDs for which the split should be carried out
    ids_current_all: list of all IDs (for ID-num mapping)
    df_cluster: dataframe columns ID, cluster_ID with cluster entries (None for random sampling disregarding any clustering information)
    train_ids_prev: IDs in the training set from previous steps (e.g. LM tuning)
    val_ids_prev: IDs in the validation set from previous steps (e.g. LM tuning)
    test_ids_prev: IDs in the test set from previous steps (e.g. LM tuning)

    sampling_ratio: desired split for train/val/test set (this is specified in terms of clusters not in terms of samples- although randomize=False tries to balance samples as well; the only exception is df_cluster=None where sampling_ratio is specified in terms of samples)
    sampling_method(_train/_valtest): -1 (predefined clusters 0:train 1:valid 2:test) 1 (cluser-based; take all members) 2 (cluster-based; take one random sequence per cluster) 3(cluster-based; use representative from clustering; in this case there might be the issue that potentially not for all clusters a representative can be picked if the clustering representative is not present)
    subsampling_ratio_train: select on subsampling_ratio_train of all assigned train clusters
    randomize: randomize remaining cluster during distribution phase/ otherwise sort by members
    random_seed: seed for sampling_method in case randomize is activated
    save_prev_ids: optionally saves previous IDs in full form (potentially required for downstream clustering tasks)

    df_cluster_redundancy: second cluster df to be used to extend the train set of the already determined split to include all cluster members
    ids_current_redundancy: additional entries to be distributed (pass e.g. ids_current)
    
    output: train_IDs.npy/val_IDs.npy/test_IDs.npy (numerical indices designating rows in tok.npy)
    train_IDs_prev.npy/val_IDs_prev.npy/test_IDs_prev.npy (original non-numerical indices for all entries that were ever assigned to the respective sets)
    these have to be passed to the call of train_test_split in the next layer (e.g. classification after training LM) to obtain a consistent split
    '''

    assert(len(np.unique(ids_current))==len(ids_current)),"Error in train_test_split: ids_current contains duplicates."
    assert(len(np.unique(ids_current_all))==len(ids_current_all)),"Error in train_test_split: ids_current_all contains duplicates."

    print("\n\nPerforming train-test split:",len(ids_current), "samples.")
    np.random.seed(random_seed)

    PATH=Path(path)
    
    # no clustering criterion for train-test-validation split- in this case our options are very limited: we can only keep existing assignments from previous
    if(df_cluster is None):
        #first step: keep existing assignments from previous steps (if any)
        test_ids_current = list(np.intersect1d(ids_current,test_ids_prev))
        val_ids_current = list(np.intersect1d(ids_current,val_ids_prev))
        train_ids_current = list(np.intersect1d(ids_current,train_ids_prev))
        ids_remaining =np.setdiff1d(np.setdiff1d(np.setdiff1d(ids_current,test_ids_current),val_ids_current),train_ids_current)
        
        if(len(train_ids_current)>0 or len(val_ids_current)>0 or len(test_ids_current)>0):
            print("sample sizes after applying previous index assignments:",len(train_ids_current),len(val_ids_current),len(test_ids_current),"total:",len(train_ids_current)+len(val_ids_current)+len(test_ids_current))
        
        #second step: do the split for the remaining indices
        ids_remaining = np.array(ids_remaining)
        #determine refined sampling ratios (considering that we already have samples from step 1)
        targets = np.maximum(0,len(ids_current)*np.array(sampling_ratio) - np.array([len(train_ids_current),len(val_ids_current),len(test_ids_current)]))
        sampling_ratio_refined = targets/np.sum(targets)

        ind_select=np.random.permutation(len(ids_remaining))
        train_new = ids_remaining[ind_select[:int(sampling_ratio_refined[0]*len(ind_select))]]
        val_new = ids_remaining[ind_select[int(sampling_ratio_refined[0]*len(ind_select)):int((sampling_ratio_refined[0]+sampling_ratio_refined[1])*len(ind_select))]]
        test_new = ids_remaining[ind_select[int((sampling_ratio_refined[0]+sampling_ratio_refined[1])*len(ind_select)):]]
            
        test_ids_current += list(test_new)
        val_ids_current += list(val_new)
        train_ids_current += list(train_new)
        
        if(subsampling_ratio_train<1.0):
            print("Subsampling train samples by factor",subsampling_ratio_train)
            ind_select = np.random.permutation(len(train_ids_current))[:int(subsampling_ratio_train*len(train_ids_current))]
            train_ids_current = list(np.array(train_ids_current)[ind_select])
        print("sample sizes after random split:",len(train_ids_current),len(val_ids_current),len(test_ids_current),"total:",len(train_ids_current)+len(val_ids_current)+len(test_ids_current))

    else: #sample using cluster information
        
        # Step -1: warn for entries that do not exist in the cluster db
        # loc with missing values https://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-deprecate-loc-reindex-listlike
        df_cluster_ids_current =  df_cluster.loc[df_cluster.index.intersection(ids_current)]
        non_existing= np.setdiff1d(ids_current,list(df_cluster_ids_current.index))
        if(len(non_existing)>0):
            print("WARNING: ",len(non_existing),"IDs are not contained in the clusters dataframe (will be discarded).\nFirst 10 missing IDs:",list(non_existing[:min(10,len(non_existing))]))

        # Step 0 fix sampling method for train/ valtest set
        if sampling_method_train == 1 or sampling_method_train == -1:
            sampling_function_train = pick_all_members_from_clusters
        elif sampling_method_train==2:
            sampling_function_train = pick_random_representatives_from_clusters
        elif sampling_method_train==3:
            sampling_function_train = pick_representative_from_clusters
        else:
            assert(False)

        if sampling_method_valtest == 1 or sampling_method_train==-1:
            sampling_function_valtest = pick_all_members_from_clusters
        elif sampling_method_valtest==2:
            sampling_function_valtest = pick_random_representatives_from_clusters
        elif sampling_method_valtest==3:
            sampling_function_valtest = pick_representative_from_clusters
        else:
            assert(False)

        # Step 1 determine all predetermined clusters (important to use the full df_cluster for lookup here)
        train_clusters = find_clusters_from_members(train_ids_prev, df_cluster)
        val_clusters = find_clusters_from_members(val_ids_prev, df_cluster)
        test_clusters = find_clusters_from_members(test_ids_prev, df_cluster)
        all_clusters = list(df_cluster_ids_current["cluster_ID"].unique())

        print("Initial samples:",len(df_cluster_ids_current), "Initial clusters:",len(all_clusters))
        # Step 2 pick from predetermined clusters
        train_ids_current = sampling_function_train(train_clusters, df_cluster_ids_current)
        val_ids_current = sampling_function_valtest(val_clusters, df_cluster_ids_current)
        test_ids_current = sampling_function_valtest(test_clusters, df_cluster_ids_current)
        #update clusters (to account for possibly unreachable initial clusters)
        train_clusters = find_clusters_from_members(train_ids_current, df_cluster_ids_current)
        val_clusters = find_clusters_from_members(val_ids_current, df_cluster_ids_current)
        test_clusters = find_clusters_from_members(test_ids_current, df_cluster_ids_current)

        if(len(train_ids_current)>0 or len(val_ids_current)>0 or len(test_ids_current)>0):
            print("train/valid/test set after exploiting existing cluster assignments:")
            print_set_summary(train_ids_current, val_ids_current, test_ids_current, train_clusters, val_clusters, test_clusters)

        # Step 3 determine potentially remaining clusters
        clusters_remaining = np.setdiff1d(np.setdiff1d(np.setdiff1d(all_clusters,train_clusters),val_clusters),test_clusters)
        # subsample train clusters (after determining clusters_remaining to avoid redistributing already assigned clusters)
        if(len(train_ids_current)>0 and subsampling_ratio_train<1.0):
            print("Subsampling existing train clusters by factor",subsampling_ratio_train)
            ind_select = np.random.permutation(len(train_clusters))[:int(subsampling_ratio_train*len(train_clusters))]
            train_clusters = list(np.array(train_clusters)[ind_select])
            train_ids_current = sampling_function_train(train_clusters, df_cluster_ids_current)
            
        if(len(clusters_remaining)>0):
            print("remaining clusters to distribute:",len(clusters_remaining))
            # Step 4 distribute the remaining clusters
            # Step 4a sort/randomize remaining clusters
            if(randomize is False):
                df_remaining=df_cluster_ids_current[df_cluster_ids_current["cluster_ID"].isin(clusters_remaining)]
                #use mergesort for stable sorting
                clusters_remaining = np.array(df_remaining.groupby("cluster_ID").size().reset_index(name='counts').sort_values("counts",ascending=False, kind='mergesort')["cluster_ID"])
            else:
                clusters_remaining = np.random.shuffle(clusters_remaining)
            
            # Step 4b determine sampling targets
            targets = np.maximum(0,len(all_clusters)*np.array(sampling_ratio) - np.array([len(train_clusters),len(val_clusters),len(test_clusters)]))
            sampling_target_refined = np.rint(len(clusters_remaining)*targets/np.sum(targets)).astype(int)
            sampling_target_refined[0] += int(len(clusters_remaining)-np.sum(sampling_target_refined))#assign final cluster to train set in case of rounding issues

            # Step 4c distribute clusters
            clusters_new = [[],[],[]]
            if(sampling_method_train==-1):#special case:predetermined clusters
                clusters_new = [[0],[1],[2]]
            else:    
                last_set = 2 #0:train 1:val 2:test

                for c in clusters_remaining:
                    #cycle through sets to ensure an approximately equal distribution in terms of samples (in case clusters are sorted by number of members)
                    if sampling_target_refined[(last_set+1)%3]>0:
                        last_set = (last_set+1)%3
                    elif sampling_target_refined[(last_set+2)%3]>0:
                        last_set = (last_set+2)%3
                    #otherwise last_set stays as it is

                    #subtract 1 from target counter
                    sampling_target_refined[last_set] -= 1
                    #append new cluster
                    clusters_new[last_set].append(c)
            #subsample train clusters
            if(subsampling_ratio_train<1.0):
                print("Subsampling newly picked train clusters by factor",subsampling_ratio_train)
                ind_select = np.random.permutation(len(clusters_new[0]))[:int(subsampling_ratio_train*len(clusters_new[0]))]
                clusters_new[0] = list(np.array(clusters_new[0])[ind_select])
            
            # Step 4d pick from these clusters
            train_ids_new = sampling_function_train(clusters_new[0], df_cluster_ids_current)
            val_ids_new = sampling_function_valtest(clusters_new[1], df_cluster_ids_current)
            test_ids_new = sampling_function_valtest(clusters_new[2], df_cluster_ids_current)
            #determine clusters from the picked member (might have missed some by picking non-existing representatives)
            train_clusters_new = find_clusters_from_members(train_ids_new, df_cluster_ids_current)
            val_clusters_new = find_clusters_from_members(val_ids_new, df_cluster_ids_current)
            test_clusters_new = find_clusters_from_members(test_ids_new, df_cluster_ids_current)

            # check for unassigned clusters (only relevant for sampling_method=3 and only if pick_first_member_instead is not set)
            if(sampling_method_train != -1 and (len(clusters_new[0])-len(train_clusters_new)>0 or len(clusters_new[1])-len(val_clusters_new)>0 or len(clusters_new[2])-len(test_clusters_new)>0)):
                train_unassigned = np.setdiff1d(clusters_new[0], train_clusters_new)
                val_unassigned = np.setdiff1d(clusters_new[1], val_clusters_new)
                test_unassigned = np.setdiff1d(clusters_new[2], test_clusters_new)
                print("INFO:",len(train_unassigned),"/",len(val_unassigned),"/",len(test_unassigned),"clusters in train/val/test set will not be represented due to missing cluster representants (e.g. due to isoforms or filtering rules).")


            #only report if newly assigned clusters and existing clusters are present
            if((len(train_ids_new)>0 or len(val_ids_new)>0 or len(test_ids_new)>0) and (len(train_ids_current)>0 or len(val_ids_current)>0 or len(test_ids_current)>0)):
                print("newly assigned entries in train/valid/test set:")
                print_set_summary(train_ids_new, val_ids_new, test_ids_new, train_clusters_new, val_clusters_new, test_clusters_new)

            # Step 5 create final index lists and output
            train_ids_current = np.concatenate([train_ids_current,train_ids_new])
            val_ids_current = np.concatenate([val_ids_current,val_ids_new])
            test_ids_current = np.concatenate([test_ids_current,test_ids_new])
            train_clusters = np.concatenate([train_clusters, train_clusters_new])
            val_clusters = np.concatenate([val_clusters, val_clusters_new])
            test_clusters = np.concatenate([test_clusters, test_clusters_new])
        else:
            print("no remaining clusters to distribute")

        print("\nfinal train/valid/test set:")
        print_set_summary(train_ids_current, val_ids_current, test_ids_current, train_clusters, val_clusters, test_clusters)
    
    #add redundant training sequences
    if(len(ids_current_redundancy)>0 and df_cluster_redundancy is not None):
        df_cluster_redundancy_ids_current =  df_cluster_redundancy.loc[df_cluster_redundancy.index.intersection(np.union1d(ids_current_redundancy, ids_current))]
        train_clusters_current = find_clusters_from_members(train_ids_current, df_cluster_redundancy_ids_current)
        train_ids_current_redundant = pick_all_members_from_clusters(train_clusters_current, df_cluster_redundancy_ids_current)
        train_ids_current = np.union1d(train_ids_current, train_ids_current_redundant)
        print("\nfinal train/valid/test set (after adding redundant training sequences):")
        print_set_summary(train_ids_current, val_ids_current, test_ids_current, train_clusters, val_clusters, test_clusters)
    
    #turn IDs into numerical IDs (consistent with ID.npy)
    ids_current_dict = {s: i for i,s in enumerate(ids_current_all)}
    
    test_ids_current_num = np.sort([ids_current_dict[a] for a in test_ids_current])
    val_ids_current_num = np.sort([ids_current_dict[a] for a in val_ids_current])
    train_ids_current_num = np.sort([ids_current_dict[a] for a in train_ids_current])
    np.save(PATH/"test_IDs.npy",test_ids_current_num)
    np.save(PATH/"val_IDs.npy",val_ids_current_num)
    np.save(PATH/"train_IDs.npy",train_ids_current_num)
    
    #save updated concatenated ID-lists
    if(save_prev_ids):
        np.save(PATH/"test_IDs_prev.npy",np.union1d(test_ids_prev,test_ids_current))
        np.save(PATH/"val_IDs_prev.npy",np.union1d(val_ids_prev,val_ids_current))
        np.save(PATH/"train_IDs_prev.npy",np.union1d(train_ids_prev,train_ids_current))

def cv_split(path, ids_current, ids_current_all, df_cluster, clusters_prev=[], nfolds=10, sampling_method_train=1, sampling_method_valtest=3, randomize=False, random_seed=42, save_prev_ids=False):
    '''performs nfold split according to clusters in df_cluster'''
    assert(len(np.unique(ids_current))==len(ids_current)),"Error in cv_split: ids_current contains duplicates."
    assert(len(np.unique(ids_current_all))==len(ids_current_all)),"Error in cv_split: ids_current_all contains duplicates."
    
    print("\n\nPerforming cv split:",len(ids_current), "samples.")

    PATH=Path(path)
    
    # if no cluster df is provided put all samples in separate clusters
    if(df_cluster is None):
        df_cluster = pd.DataFrame(ids_current_all, ids_current_all)
        df_cluster.columns=["cluster_ID"]

    # Step -1: warn for entries that do not exist in the cluster db
    # loc with missing values https://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-deprecate-loc-reindex-listlike
    df_cluster_ids_current =  df_cluster.loc[df_cluster.index.intersection(ids_current)]
    non_existing= np.setdiff1d(ids_current,list(df_cluster_ids_current.index))
    if(len(non_existing)>0):
        print("WARNING: ",len(non_existing),"IDs are not contained in the clusters dataframe (will be discarded).\nFirst 10 missing IDs:",list(non_existing[:min(10,len(non_existing))]))

    # Step 0 fix sampling method for train/ valtest set
    if sampling_method_train == 1 or sampling_method_train == -1:
        sampling_function_train = pick_all_members_from_clusters
    elif sampling_method_train==2:
        sampling_function_train = pick_random_representatives_from_clusters
    elif sampling_method_train==3:
        sampling_function_train = pick_representative_from_clusters
    else:
        assert(False)

    if sampling_method_valtest == 1 or sampling_method_train==-1:
        sampling_function_valtest = pick_all_members_from_clusters
    elif sampling_method_valtest==2:
        sampling_function_valtest = pick_random_representatives_from_clusters
    elif sampling_method_valtest==3:
        sampling_function_valtest = pick_representative_from_clusters
    else:
        assert(False)

    # Step 1 determine all predetermined clusters (important to use the full df_cluster for lookup here)
    if(len(clusters_prev)>0):
        print("number of predetermined clusters:")
        print([len(c) for c in clusters_prev])
    all_clusters_prev = [item for sublist in clusters_prev for item in sublist] if len(clusters_prev)>0 else []
    all_clusters = list(df_cluster_ids_current["cluster_ID"].unique())

    # Step 2 determine potentially remaining clusters
    clusters_remaining = np.setdiff1d(all_clusters,all_clusters_prev)
    df_remaining=df_cluster_ids_current[df_cluster_ids_current["cluster_ID"].isin(clusters_remaining)]
    splits_remaining = split_clusters_nfolds(df_remaining, nfolds=nfolds, randomize=randomize, random_seed=random_seed)

    if(len(clusters_prev)==0):
        clusters_present = splits_remaining
    else:
        clusters_present = [p+r for p,r in zip(clusters_prev,splits_remaining)]
    print("number of final clusters:")
    print([len(c) for c in clusters_present])
    if(save_prev_ids):
        np.save(PATH/"cluster_IDs_CV_prev.npy",clusters_present)

    # Step 3 actually pick samples (use one fold for valid, one for test and the rest for train)
    train_ids = []
    val_ids = []
    test_ids = []

    for i in range(nfolds):
        train_clusters = []
        for j in range(nfolds):
            if(j!=i and j!=((i-1)%nfolds)):
                train_clusters = train_clusters + clusters_present[j]
        val_clusters = clusters_present[i]
        test_clusters = clusters_present[(i-1)%nfolds]

        train_samples = sampling_function_train(train_clusters,df_cluster_ids_current)
        val_samples = sampling_function_valtest(val_clusters,df_cluster_ids_current)
        test_samples = sampling_function_valtest(test_clusters,df_cluster_ids_current)
        train_ids.append(train_samples)
        val_ids.append(val_samples)
        test_ids.append(test_samples)
    
        print("\ntrain/valid/test fold:",i)
        print_set_summary(train_samples, val_samples, test_samples, train_clusters, val_clusters, test_clusters)
    
    #turn IDs into numerical IDs  using ids_current_all (consistent with ID.npy)
    ids_current_dict = {s: i for i,s in enumerate(ids_current_all)}
    
    val_ids_num = [np.sort([ids_current_dict[a] for a in v]) for v in val_ids]
    test_ids_num = [np.sort([ids_current_dict[a] for a in v]) for v in test_ids]
    train_ids_num = [np.sort([ids_current_dict[a] for a in t]) for t in train_ids]
    
    np.save(PATH/"val_IDs_CV.npy",val_ids_num)
    np.save(PATH/"test_IDs_CV.npy",test_ids_num)
    np.save(PATH/"train_IDs_CV.npy",train_ids_num)
