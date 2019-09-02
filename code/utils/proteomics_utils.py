from lxml import etree
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split

import Bio
from Bio import SeqIO

#pssm libs
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from Bio.Alphabet import ProteinAlphabet
from Bio.Blast.Applications import NcbipsiblastCommandline
from multiprocessing import Pool

from pathlib import Path

#console
from tqdm import tqdm as tqdm
import re
import os
import itertools
#jupyter
#from tqdm import tqdm_notebook as tqdm
#not supported in current tqdm version
#from tqdm.autonotebook import tqdm

#import logging
#logging.getLogger('proteomics_utils').addHandler(logging.NullHandler())
#logger=logging.getLogger('proteomics_utils')

#for cd-hit
import subprocess

from sklearn.metrics import f1_score

import hashlib #for mhcii datasets

from utils.dataset_utils import split_clusters_single,pick_all_members_from_clusters

#######################################################################################################
#Parsing all sorts of protein data
#######################################################################################################
def parse_uniprot_xml(filename,max_entries=0,parse_features=[]):
    '''parse uniprot xml file, which contains the full uniprot information (e.g. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz)
    using custom low-level https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    c.f. for full format https://www.uniprot.org/docs/uniprot.xsd

    parse_features: a list of strings specifying the kind of features to be parsed such as "modified residue" for phosphorylation sites etc. (see https://www.uniprot.org/help/mod_res)
        (see the xsd file for all possible entries)
    '''
    context = etree.iterparse(str(filename), events=["end"], tag="{http://uniprot.org/uniprot}entry")
    context = iter(context)
    rows =[]
    
    for _, elem in tqdm(context):
        parse_func_uniprot(elem,rows,parse_features=parse_features)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)

    return df

def parse_func_uniprot(elem, rows, parse_features=[]):
    '''extracting a single record from uniprot xml'''
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    sequence=""
    #print(seqs)
    for s in seqs:
        sequence=s.text
        #print("sequence",sequence)
        if sequence =="" or str(sequence)=="None":
            continue
        else:
            break
    #Sequence & fragment
    sequence=""
    fragment_map = {"single":1, "multiple":2}
    fragment = 0
    seqs = elem.findall("{http://uniprot.org/uniprot}sequence")
    for s in seqs:
        if 'fragment' in s.attrib:
            fragment = fragment_map[s.attrib["fragment"]]
        sequence=s.text
        if sequence != "":
            break
    #print("sequence:",sequence)
    #print("fragment:",fragment)

    #dataset
    dataset=elem.attrib["dataset"]

    #accession
    accession = ""
    accessions = elem.findall("{http://uniprot.org/uniprot}accession")
    for a in accessions:
        accession=a.text
        if accession !="":#primary accession! https://www.uniprot.org/help/accession_numbers!!!
            break
    #print("accession",accession)

    #protein existence (PE in plain text)
    proteinexistence_map = {"evidence at protein level":5,"evidence at transcript level":4,"inferred from homology":3,"predicted":2,"uncertain":1}
    proteinexistence = -1
    accessions = elem.findall("{http://uniprot.org/uniprot}proteinExistence")
    for a in accessions:
        proteinexistence=proteinexistence_map[a.attrib["type"]]
        break
    #print("protein existence",proteinexistence)

    #name
    name = ""
    names = elem.findall("{http://uniprot.org/uniprot}name")
    for n in names:
        name=n.text
        break
    #print("name",name)

    #organism
    organism = ""
    organisms = elem.findall("{http://uniprot.org/uniprot}organism")
    for s in organisms:
        s1=s.findall("{http://uniprot.org/uniprot}name")
        for s2 in s1:
            if(s2.attrib["type"]=='scientific'):
                organism=s2.text
                break
        if organism !="":
            break
    #print("organism",organism)

    #dbReference: PMP,GO,Pfam, EC
    ids = elem.findall("{http://uniprot.org/uniprot}dbReference")
    pfams = []
    gos =[]
    ecs = []
    pdbs =[]
    for i in ids:
        #print(i.attrib["id"],i.attrib["type"])

        #cf. http://geneontology.org/external2go/uniprotkb_kw2go for Uniprot Keyword<->GO mapping
        #http://geneontology.org/ontology/go-basic.obo for List of go terms
        #https://www.uniprot.org/help/keywords_vs_go keywords vs. go
        if(i.attrib["type"]=="GO"):
            tmp1 = i.attrib["id"]
            for i2 in i:
                if i2.attrib["type"]=="evidence":
                    tmp2= i2.attrib["value"]
            gos.append([int(tmp1[3:]),int(tmp2[4:])]) #first value is go code, second eco evidence ID (see mapping below)
        elif(i.attrib["type"]=="Pfam"):
            pfams.append(i.attrib["id"])
        elif(i.attrib["type"]=="EC"):
            ecs.append(i.attrib["id"])
        elif(i.attrib["type"]=="PDB"):
            pdbs.append(i.attrib["id"])
    #print("PMP: ", pmp)
    #print("GOs:",gos)
    #print("Pfams:",pfam)
    #print("ECs:",ecs)
    #print("PDBs:",pdbs)


    #keyword
    keywords = elem.findall("{http://uniprot.org/uniprot}keyword")
    keywords_lst = []
    #print(keywords)
    for k in keywords:
        keywords_lst.append(int(k.attrib["id"][-4:]))#remove the KW-
    #print("keywords: ",keywords_lst)

    #comments = elem.findall("{http://uniprot.org/uniprot}comment")
    #comments_lst=[]
    ##print(comments)
    #for c in comments:
    #    if(c.attrib["type"]=="function"):
    #        for c1 in c:
    #            comments_lst.append(c1.text)
    #print("function: ",comments_lst)

    #ptm etc
    if len(parse_features)>0:
        ptms=[]
        features = elem.findall("{http://uniprot.org/uniprot}feature")
        for f in features:
            if(f.attrib["type"] in parse_features):#only add features of the requested type
                locs=[]
                for l in f[0]:
                    locs.append(int(l.attrib["position"]))
                ptms.append([f.attrib["type"],f.attrib["description"] if 'description' in f.attrib else "NaN",locs, f.attrib['evidence'] if 'evidence' in f.attrib else "NaN"])
        #print(ptms)

    data_dict={"ID": accession, "name": name, "dataset":dataset, "proteinexistence":proteinexistence, "fragment":fragment, "organism":organism, "ecs": ecs, "pdbs": pdbs, "pfams" : pfams, "keywords": keywords_lst, "gos": gos,  "sequence": sequence}

    if len(parse_features)>0:
        data_dict["features"]=ptms
    #print("all children:")
    #for c in elem:
    #    print(c)
    #    print(c.tag)
    #    print(c.attrib)
    rows.append(data_dict)

def parse_uniprot_seqio(filename,max_entries=0):
    '''parse uniprot xml file using the SeqIO parser (smaller functionality e.g. does not extract evidence codes for GO)'''
    sprot = SeqIO.parse(filename, "uniprot-xml")
    rows = []
    for p in tqdm(sprot):
        accession = str(p.name)
        name = str(p.id)
        dataset = str(p.annotations['dataset'])
        organism = str(p.annotations['organism'])
        ecs, pdbs, pfams, gos = [],[],[],[]
        
        for ref in p.dbxrefs:
            k = ref.split(':')
            if k[0] == 'GO':
                gos.append(':'.join(k[1:]))
            elif k[0] == 'Pfam':
                pfams.append(k[1])
            elif k[0] == 'EC':
                ecs.append(k[1])
            elif k[0] == 'PDB':
                pdbs.append(k[1:])
        if 'keywords' in p.annotations.keys():
            keywords = p.annotations['keywords']
        else:
            keywords = []
        
        sequence = str(p.seq)
        row = {
            'ID': accession, 
            'name':name, 
            'dataset':dataset,
            'organism':organism,
            'ecs':ecs,
            'pdbs':pdbs,
            'pfams':pfams,
            'keywords':keywords,
            'gos':gos,
            'sequence':sequence}
        rows.append(row)
        if(max_entries>0 and len(rows)==max_entries):
            break          

    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)

    return df
    
def filter_human_proteome(df_sprot):
    '''extracts human proteome from swissprot proteines in DataFrame with column organism '''
    is_Human = np.char.find(df_sprot.organism.values.astype(str), "Human") !=-1
    is_human = np.char.find(df_sprot.organism.values.astype(str), "human") !=-1
    is_sapiens = np.char.find(df_sprot.organism.values.astype(str), "sapiens") !=-1
    is_Sapiens = np.char.find(df_sprot.organism.values.astype(str), "Sapiens") !=-1
    return df_sprot[is_Human|is_human|is_sapiens|is_Sapiens]

def filter_aas(df, exclude_aas=["B","J","X","Z"]):
    '''excludes sequences containing exclude_aas: B = D or N, J = I or L, X = unknown, Z = E or Q'''
    return df[~df.sequence.apply(lambda x: any([e in x for e in exclude_aas]))]

#######################################################################################################
def parse_uniprot_keywords(filename):
    '''parsing uniprot keywords (keyword->description) ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/docs/keywlist.txt'''
    descr_list=[]
    descr_found=False
    with open(filename, 'r') as f:
        for l in f:
            if l[:2]=="ID":
                descr=l[5:-2]
                descr_found = True
            elif descr_found is True:
                descr_found = False
                kw=int(l[-5:])
                descr_list.append({"KW":kw,"descr":descr})

    df_keywlist = pd.DataFrame(descr_list).set_index("KW")
    #df_keywlist = df_keywlist.set_index("KW")
    return df_keywlist

######################################################################################################
def trace_go_parents(goid,go_terms):
    '''trace go term and all parents recursively'''
    if goid in go_terms.index:
        parents = go_terms[go_terms.index == goid].iloc[0].is_a
        if(len(parents)==0):
            return [goid]
        else:
            res = [goid]
            for p in parents:
                res += trace_go_parents(p,go_terms)
            return list(set(res))
    else:
        return []

def int_to_go(goint):
    '''convert integer to GO-string'''
    return "GO:"+format(goint, '07d')

def go_to_int(go):
    '''convert GO-string to integer'''
    return int(go[3:])

def parse_go_terms(filename):
    '''parsing GO terms (GO->is_a,name,namespace,part_of,regulates) http://geneontology.org/ontology/go-basic.obo'''
    descr_list=[]
    
    with open(filename, 'r') as f:
        id=-1
        name=""
        namespace = ""
        is_a =[]
        part_of=[]
        regulates=[]
        id_found=False        
        
        for l in f:
            #print(l)
            if l[:7]=="id: GO:":
                id=int(l[7:-1])
                id_found = True
            elif id_found is True:
                if l[:5]=="name:":
                    name = l[6:-1]
                elif l[:10]=="namespace:":
                    namespace = l[11:-1]
                elif l[:25]=="relationship: part_of GO:":
                    part_of.append(int(l[25:32]))
                elif l[:27]=="relationship: regulates GO:":
                    regulates.append(int(l[27:34]))
                elif l[:9]=="is_a: GO:":
                    is_a.append(int(l[9:16]))       
            if l[:6] == "[Term]":
                #flushing based on [Term] here relies on the fact that the last entry in the file is empty
                if id_found is True:
                    descr_list.append({"GO":id, "name":name, "namespace":namespace, "is_a":is_a, "part_of":part_of, "regulates":regulates})
                id=-1
                name=""
                namespace = ""
                is_a =[]
                part_of=[]
                regulates=[]
                id_found = False

    df_keywlist = pd.DataFrame(descr_list).set_index("GO")
    return df_keywlist

def generate_deeprotein_dataset(train_on_cafa3_original=False, eval_on_cafa3_test=True,go_selection=[],discard_empty=True,filename_train_cafa3_original="../data/train_cafa3_original.csv",filename_train_cafa3_expanded="../data/train_cafa3_expanded.csv", filename_test_cafa3="../data/test_cafa3.csv", filename_test_cafa3_deepgo_comparison="../data/test_cafa3_deepgo_comparison.csv"):
    '''generates the deeprotein GO classification dataset based on the output of the scripts from https://github.com/juzb/DeeProtein (download.sh datasets.sh datasets_up.sh)'''
    def read_deeprotein_csv(filename,gos=[]):
        print("Loading GO data from",filename)
        df = pd.read_csv(filename,sep=";",header=None)
        df.columns = ["ID","sequence","GO"] if len(df.columns)==3 else ["ID","sequence","GO","xxx"]
        df = df[["ID","sequence","GO"]]
        df.GO=df.GO.apply(lambda x: x.split(","))
        if(len(gos)>0):#filter by gos
            df.GO=df.GO.apply(lambda x: [y for y in x if y in gos])
        df["label"]=df.GO
        return df.set_index("ID")
        
    if(train_on_cafa3_original):
        df_train = read_deeprotein_csv(filename_train_cafa3_original, go_selection)
    else:
        df_train = read_deeprotein_csv(filename_train_cafa3_expanded, go_selection)
    df_train["cluster_ID"]=0
    
    if(eval_on_cafa3_test):
        df_test = read_deeprotein_csv(filename_test_cafa3, go_selection)
    else:
        df_test = read_deeprotein_csv(filename_test_cafa3_deepgo_comparison, go_selection)
    df_test["cluster_ID"]=1
    
    
    print("train",len(df_train),len(np.unique(df_train.index)))
    print("test",len(df_test),len(np.unique(df_test.index)))
    
    #remove duplicates from train
    df_train = df_train[~df_train.index.isin(np.intersect1d(np.array(df_train.index),np.array(df_test.index)))]
    print("after removing train overlap with test", len(df_train),len(np.unique(df_train.index)))
    df_train = df_train.loc[~df_train.index.duplicated(keep='first')]
    print("after removing train duplicates", len(df_train),len(np.unique(df_train.index)))
    
    df = pd.concat([df_train,df_test])
        
    print("all",len(df),len(np.unique(df.index)))
    if(discard_empty is True):
        df = df[df["label"].apply(lambda x:len(x)) >0]
        print("all after discarding empties",len(df),len(np.unique(df.index)))
    
    return df

def go_predictions_to_deepgo_pkl(preds,labels,ids,label_itos,df_itos_filename="lbl_itos.pkl",df_preds_filename="preds.pkl"):
    '''converts predictions into deepgo format that can be processed by deeprotein/deepgo evaluation scripts
    call: python "$CODE_DIR"/DeeProtein/scripts/deep_go_eval.py preds.pkl go-basic.obo  lbl_itos.pkl > deep_go_eval.txt
    '''
    #itos
    df_itos=pd.DataFrame(label_itos)
    df_itos.columns = ["functions"]
    df_itos.to_pickle(df_itos_filename)
    #preds
    rows=[]
    for pr,lbl,id in zip(preds,labels,ids):
        gos = set([label_itos[i] for i in np.where(lbl==1)[0]])
        rows.append({"gos":gos,"labels":lbl,"targets":id,"predictions":pr})
    df_preds = pd.DataFrame(rows)
    df_preds.to_pickle(df_preds_filename)
    return df_itos,df_preds
######################################################################################################
def explode_clusters_df(df_cluster):
    '''aux. function to convert cluster dataframe from one row per cluster to one row per ID'''
    df=df_cluster.reset_index(level=0)
    rows = []
    if('repr_accession' in df.columns):#include representative if it exists
        _ = df.apply(lambda row: [rows.append([nn,row['entry_id'], row['repr_accession']==nn ]) for nn in row.members], axis=1)
        df_exploded = pd.DataFrame(rows, columns=['ID',"cluster_ID","representative"]).set_index(['ID'])
    else:
        _ = df.apply(lambda row: [rows.append([nn,row['entry_id']]) for nn in row.members], axis=1)
        df_exploded = pd.DataFrame(rows, columns=['ID',"cluster_ID"]).set_index(['ID'])
    return df_exploded

def parse_uniref(filename,max_entries=0,parse_sequence=False, df_selection=None, exploded=True):
    '''parse uniref (clustered sequences) xml ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/uniref50.xml.gz unzipped 100GB file
    using custom low-level parser https://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    max_entries: only return first max_entries entries (0=all)
    parse_sequences: return also representative sequence
    df_selection: only include entries with accessions that are present in df_selection.index (None keeps all records)
    exploded: return one row per ID instead of one row per cluster
    c.f. for full format ftp://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref50/README
    '''
    #issue with long texts https://stackoverflow.com/questions/30577796/etree-incomplete-child-text
    #wait for end rather than start tag
    context = etree.iterparse(str(filename), events=["end"], tag="{http://uniprot.org/uniref}entry")
    context = iter(context)
    rows =[]
    
    for _, elem in tqdm(context):
        parse_func_uniref(elem,rows,parse_sequence=parse_sequence, df_selection=df_selection)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("entry_id")
    df["num_members"]=df.members.apply(len)

    if(exploded):
        return explode_clusters_df(df)
    return df

def parse_func_uniref(elem, rows, parse_sequence=False, df_selection=None):
    '''extract a single uniref entry'''
    #entry ID
    entry_id = elem.attrib["id"]
    #print("cluster id",entry_id)
    
    #name
    name = ""
    names = elem.findall("{http://uniprot.org/uniref}name")
    for n in names:
        name=n.text[9:]
        break
    #print("cluster name",name)
    
    members=[]
    #representative member
    repr_accession = ""
    repr_sequence =""
    repr = elem.findall("{http://uniprot.org/uniref}representativeMember")
    for r in repr:
        s1=r.findall("{http://uniprot.org/uniref}dbReference")
        for s2 in s1:
            for s3 in s2:
                if s3.attrib["type"]=="UniProtKB accession":
                    if(repr_accession == ""):
                        repr_accession = s3.attrib["value"]#pick primary accession
                    members.append(s3.attrib["value"])
        if parse_sequence is True:
            s1=r.findall("{http://uniprot.org/uniref}sequence")
            for s2 in s1:
                repr_sequence = s2.text
                if repr_sequence !="":
                    break
        
    #print("representative member accession:",repr_accession)
    #print("representative member sequence:",repr_sequence)
    
    #all members
    repr = elem.findall("{http://uniprot.org/uniref}member")
    for r in repr:
        s1=r.findall("{http://uniprot.org/uniref}dbReference")
        for s2 in s1:
            for s3 in s2:
                if s3.attrib["type"]=="UniProtKB accession":
                    members.append(s3.attrib["value"]) #add primary and secondary accessions
    #print("members", members)

    if(not(df_selection is None)): #apply selection filter
        members = [y for y in members if y in df_selection.index]
    #print("all children")
    #for c in elem:
    #    print(c)
    #    print(c.tag)
    #    print(c.attrib)
    if(len(members)>0):
        data_dict={"entry_id": entry_id, "name": name, "repr_accession":repr_accession, "members":members}
        if parse_sequence is True:
            data_dict["repr_sequence"]=repr_sequence
        rows.append(data_dict)
###########################################################################################################################
#preprocessing for homology dataset
###########################################################################################################################

def generate_homology_scop(filename_postfix="a.1.1.2", data_dir='.',split_ratio_train=None,cdhit_threshold=0.5,cdhit_alignment_coverage=0.8):
    '''prepares df from http://www.bioinf.jku.at/software/LSTM_protein/''' 
    data_dir = Path(data_dir)
    file_neg_test = data_dir/("neg-test."+filename_postfix+".fasta")
    file_pos_test = data_dir/("pos-test."+filename_postfix+".fasta")
    file_neg_train = data_dir/("neg-train."+filename_postfix+".fasta")
    file_pos_train = data_dir/("pos-train."+filename_postfix+".fasta")

    df_neg_test = fasta_to_df(str(file_neg_test))
    df_neg_test["label"]= 0
    df_neg_test["cluster_ID"]=1 if split_ratio_train is None else 2

    df_pos_test = fasta_to_df(str(file_pos_test))
    df_pos_test["label"]= 1
    df_pos_test["cluster_ID"]=1 if split_ratio_train is None else 2

    df_neg_train = fasta_to_df(str(file_neg_train))
    df_neg_train["label"]= 0
    df_neg_train["cluster_ID"]=0

    df_pos_train = fasta_to_df(str(file_pos_train))
    df_pos_train["label"]= 1
    df_pos_train["cluster_ID"]=0

    df=pd.concat([df_neg_test, df_pos_test, df_neg_train, df_pos_train])
    #take care of duplicate IDs
    df["IDcount"]=df.groupby('ID', as_index=False).cumcount()
    df["ID"]= df.apply(lambda x:x["ID"] if x["IDcount"]==0 else x["ID"]+"variant"+str(x["IDcount"]),axis=1)
    df=df.set_index("ID")

    if(split_ratio_train is not None):#split redundant train into train+val
        df_train = df[df.cluster_ID==0].copy()
        df_train_cluster = clusters_df_from_sequence_df(df_train,threshold=cdhit_threshold,alignment_coverage=cdhit_alignment_coverage)
        print([split_ratio_train,1-split_ratio_train],len([split_ratio_train,1-split_ratio_train]))
        splits = split_clusters_single(df_train_cluster, [split_ratio_train,1-split_ratio_train])
        splits_val_ids =  pick_all_members_from_clusters(splits[1], df_train_cluster)
        df.loc[df.index.isin(splits_val_ids),"cluster_ID"] = 1
        
    return df

###########################################################################################################################
#extracting truncated ECs from feature dataframe
###########################################################################################################################
def truncate_ec(ec,level=1):
    ''''truncate a single EC class (as string) to a given level'''
    if level==0:#level=0: EC/NoEC classification- always return EC as this function will only be called for ECs
        return "EC"
    ec_sep=ec.split(".")[:level]
    ec_trunc=ec_sep[0]
    for e in ec_sep[1:]:
        ec_trunc+="."+e
    return ec_trunc

def truncate_ecs(ec_lst,level=1,drop_incomplete=True):
    '''
    truncates a list of ecs, drops incomplete entries (after truncation), and inserts NoEC
    '''
    if(len(ec_lst)==0 or (len(ec_lst)==1 and ec_lst[0]=="NoEC")):
        return ["NoEC"]
    if(drop_incomplete):
        tmp=[x for x in [truncate_ec(y,level) for y in ec_lst] if not("-" in x)]
    else:
        tmp=[truncate_ec(y,level) for y in ec_lst]
    return list(np.unique(tmp))

def ecs_from_df(df, level=1, include_NoEC=False, drop_incomplete=True, drop_ec7=True):
    '''map ecs (truncated to a certain level) to integers
    input is a ec dataframe with sequence and ecs columns (for example a uniprot dataframe or a dataframe produced by one of the methods below)
    '''
    if(not("ecs_truncated" in df.columns)):    
        df["ecs_truncated"] = df.ecs.apply(lambda x: truncate_ecs(x, level=level, drop_incomplete=drop_incomplete))
    
    column_list = ["sequence","ecs_truncated"]
    if("fragment" in df.columns):
        column_list.append("fragment")
    if("proteinexistence" in df.columns):
        column_list.append("proteinexistence")
        
    df_loc=df[column_list].copy()

    #print("include_NoEC",include_NoEC,"len",df_loc.ecs_truncated.apply(lambda x:x[0]).value_counts())
    #df_loc.to_pickle("df_loc.pkl")
    if(include_NoEC is False):#drop NoECS
        df_loc.drop(df_loc[df_loc.ecs_truncated.apply(lambda x: "NoEC" in x)].index,inplace=True)
    #print("include_NoEC",include_NoEC,"len",df_loc.ecs_truncated.apply(lambda x:x[0]).value_counts())
    
    if(drop_ec7 and level>0):#drop ec class 7 if desired
        df_loc.ecs_truncated = df_loc.ecs_truncated.apply(lambda x: [y for y in x if y[0]!="7"])
    
    df_loc.drop(df_loc[df_loc.ecs_truncated.apply(len) == 0].index,inplace=True) #drop potentially empty ECs from incomplete/ec7 entries
        
    ec_itos=list(np.unique(np.concatenate(df_loc.ecs_truncated.values)))
    ec_stoi={s:i for i,s in enumerate(ec_itos)}

    df_loc.ecs_truncated=df_loc.ecs_truncated.apply(lambda x: [ec_stoi[y] for y in x])
    
    return df_loc, ec_itos

def ecs_from_uniprot(df_uniprot, level, drop_fragments, drop_incomplete, include_NoEC, minproteinexistence_NoEC=4):
    '''creates ec dataframe from uniprot dataframe'''
    # 1. filter multiple ec-numbers
    df_uniprot = df_uniprot[df_uniprot.ecs.apply(len) <= 1]
    # 2. filter fragments
    if drop_fragments:
        df_uniprot = df_uniprot[df_uniprot.fragment == 0]
    # 3. filter non enzymes
    if not include_NoEC:
        df_uniprot = df_uniprot[(df_uniprot.ecs.apply(len) > 0)]
    else:
        df_uniprot = df_uniprot[(df_uniprot.ecs.apply(len) > 0) | (df_uniprot.proteinexistence >= minproteinexistence_NoEC)]
    # 4. truncate ecs (potentially empty lists will be dropped in ecs_from_df
    df_uniprot["ecs_truncated"] = df_uniprot.ecs.apply(lambda x: truncate_ecs(x, level=level, drop_incomplete=drop_incomplete))
    
    return df_uniprot[["sequence","fragment","proteinexistence","ecs_truncated"]]


def ecs_from_knna(filename="suppa.txt"):
    '''prepare ec dataframe from KNN dataset A'''
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
        
    tmp_seqs = []
    tmp_seq = ''
    classes = {}
    for i, row in tqdm(enumerate(content)):
        if 'EC.' in row:
            # store previous sequence before searching for new class sequences
            if len(tmp_seqs) > 0:
                tmp_seqs.append((tmp_seq_id,tmp_seq))
                tmp_seq = ''
                classes[str(ec1)] = tmp_seqs
                tmp_seqs = []
            
            # skip intro before each new class
            m = re.match('\(EC.\d\)\s\d*\s', row)
            if m:
                #print(i,row)
                splits = m.group().split(' ')
                #print(m.group())
                tmp_class, _ = splits[0], int(splits[1])
                ec_splits = tmp_class.split('.')
                
                ec1 = int(ec_splits[1][0])
        else:
            m = re.match('\(\d\)', row)
            if m and int(m.group()[1]) == 2:
                if len(tmp_seqs) > 0:
                    tmp_seqs.append((tmp_seq_id,tmp_seq))
                    tmp_seq = ''
                    classes[str(ec1)] = tmp_seqs
                    tmp_seqs = []
                ec1 = 0
            if '>' in row:
                if len(tmp_seq) > 0:
                    tmp_seqs.append((tmp_seq_id,tmp_seq))
                tmp_seq = ''
                tmp_seq_id = row.split('>')[1]
            else:
                s = ''.join([i for i in row if not i.isdigit()])
                if s.isupper(): # this needs to be checked because somewhere is a bug -.-
                    tmp_seq += s            
    tmp_seqs.append((tmp_seq_id,tmp_seq))
    classes[str(ec1)] = tmp_seqs            
    
    rows = []
    
    for key in classes.keys():
        for id, seq in classes[key]:
            rows.append([id,[key] if key!="0" else [],seq])
    df = pd.DataFrame(rows, columns=['id', 'ecs', 'sequence'])
    df = df.set_index('id')
    return df

def ecs_from_knnb(filename="suppb.txt"):
    '''prepare ec dataset from KNN dataset B'''
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    
    tmp_seqs = []
    tmp_seq = ''
    classes = {}

    for i, row in tqdm(enumerate(content)):
        if 'EC.' in row:
            # store previous sequence before searching for new class sequences
            if len(tmp_seqs) > 0:
                tmp_seqs.append((tmp_seq_id,tmp_seq))
                tmp_seq = ''
                classes[str(ec1)+"."+str(ec2)] = tmp_seqs
                tmp_seqs = []
            
            # skip intro before each new class
            m = re.match('\(EC.\d.\d*\)\s\d*\s', row)
            if m:
                #print(i,row)
                splits = m.group().split(' ')
                tmp_class, _ = splits[0], int(splits[1])
                ec_splits = tmp_class.split('.')
                ec1, ec2 = int(ec_splits[1]), int(ec_splits[2][:-1])
        else:
            if '>' in row:
                if len(tmp_seq) > 0:
                    tmp_seqs.append((tmp_seq_id,tmp_seq))
                tmp_seq = ''
                tmp_seq_id = row.split('>')[1]
            else:
                s = ''.join([i for i in row if not i.isdigit()])
                if s.isupper(): # this needs to be checked because somewhere is a bug -.-
                    tmp_seq += s
                
    tmp_seqs.append((tmp_seq_id,tmp_seq))
    classes[str(ec1)+"."+str(ec2)] = tmp_seqs            


    rows = []
    for key in classes.keys():
        splits = key.split('.')
        ec1,ec2 = int(splits[0]), int(splits[1])
        for id, seq in classes[key]:
            rows.append([id,[".".join([str(ec1), str(ec2)])],seq])
    df = pd.DataFrame(rows, columns=['id', 'ecs', 'sequence'])
    df = df.set_index('id')
    return df

def ecs_from_knn_new(df_uniprot, level=1, threshold=.4, alignment_coverage=.0, min_length=50, max_length=5000, drop_fragments=True, drop_incomplete=True, include_NoEC=True, single_label=True):
    '''mimics procedure to produce DeePre s dataset KNN_NEW'''
    
    ids, cluster_ids, reprs = [],[],[]

    print("Total number of proteins: " + str(len(df_uniprot)))
    df_filtered = ecs_from_uniprot(df_uniprot, level=level, min_length=min_length, max_length=max_length, drop_fragments=drop_fragments, drop_incomplete=drop_incomplete,include_NoEC=include_NoEC)
    print("Number of proteins after Filtering: " + str(len(df_filtered)))
    df_enzymes = df_filtered[df_filtered.ecs_truncated.apply(lambda x: x[0]) != "NoEC"]
    print("Enzymes before clustering:",len(df_enzymes))
    if level==0 or (include_NoEC):
        df_nonenzymes = df_filtered[df_filtered.ecs_truncated.apply(lambda x: x[0]) == "NoEC"]
        print("Non-enzymes before clustering:",len(df_nonenzymes))
   
    # cluster enzymes
    enzymes_clustering = clusters_df_from_sequence_df(df_enzymes, threshold=threshold, alignment_coverage=alignment_coverage, verbose=False, exploded=True)
    df_enzymes_reduced = df_enzymes[enzymes_clustering.representative]

    # cluster non-enzymes
    nonenzymes_clustering = clusters_df_from_sequence_df(df_nonenzymes, threshold=threshold, alignment_coverage=alignment_coverage, verbose=False, exploded=True)
    df_nonenzymes_reduced = df_nonenzymes[nonenzymes_clustering.representative.values]

    print("Enzymes after clustering:",len(df_enzymes_reduced))
    print("Non-enzymes after clustering:",len(df_nonenzymes_reduced))

    df_reduced = pd.concat([df_enzymes_reduced, df_nonenzymes_reduced], axis=0)
    df_cluster = pd.concat([enzymes_clustering, nonenzymes_clustering], axis=0)

    print("Total number after clustering:",len(df_reduced))
    return df_reduced[["sequence", "ecs_truncated"]], df_filtered[["sequence", "ecs_truncated"]], df_cluster

#assert(include_NoEC)
def prepare_ecpred_data(path_ec_ecpred, ec_class=1, full_train=True, train_val_ratio=1, random_seed=42):
    '''prepare dataset using ecpred accessions
        ec_class: use test-set for ec class ec_class
        full_train: use train data from all ec classes for training (this will train a binary classifier for level 0 and a 6+1d classifier for level 1- NoEC has to be always enabled)
        train_val_ratio: use train_val_ratio % for train, the rest for validation set
        '''
    #not test set counts match exactly train sets are a little bit larger than their reported sizes
    df=pd.read_pickle(path_ec_ecpred)
    df = df.set_index("accession")
    eclevel1_dict = {"Oxidoreductases":1, "Transferases":2, "Hydrolases":3, "Lyases":4, "Isomerases":5, "Ligases":6}
    df["mainclass_set"]= df.mainclass_set.apply(lambda x: eclevel1_dict[x])
    df = df.rename(columns={"traintest":"cluster_ID","ec":"ecs"})

    if(train_val_ratio<1):
        dfs_train=[]
        for i in range(1,7):
            df_tmp = df[(df.cluster_ID==0) & (df.mainclass_set==i)]
            if(i!=ec_class):
                dfs_train.append(df_tmp)
            else:
                np.random.seed(random_seed)
                tmp_ids = np.random.permutation(np.unique(df_tmp.index))
                
                dfs_train.append(df_tmp[df_tmp.index.isin(tmp_ids[:int(train_val_ratio*len(df_tmp))])])
                df_val = df_tmp[df_tmp.index.isin(tmp_ids[int(train_val_ratio*len(df_tmp)):])].copy()
                df_val["cluster_ID"]=1
        df_train = pd.concat(dfs_train)
        #print(df_train.shape)
        #print("deleted val duplicates in train:",len(np.intersect1d(np.array(df_train.index),np.array(df_val.index))))
        #df_train = df_train.drop(np.intersect1d(np.array(df_train.index),np.array(df_val.index))) #delete all duplicates from train that are already contained in val
        df_train = df_train[~df_train.index.isin(np.intersect1d(np.array(df_train.index),np.array(df_val.index)))]
        #print(df_train.shape)
        df_train = pd.concat([df_train,df_val])
    else:
        df_train = df[df.cluster_ID==0]

    df_test = df[(df.cluster_ID==1) & (df.mainclass_set==ec_class)].copy()
    
    if(full_train is True):#use concatenated train sets as train        
        if(train_val_ratio<1):
            df_test["cluster_ID"]=2
        #print("Dropping duplicates train size before:",len(df_train))
        #df_train = df_train.drop_duplicates()
        df_train = df_train.loc[~df_train.index.duplicated(keep='first')]
        #print("train size after:", len(df_train))
        #print("unique ids:",len(np.unique(df_train.index)))
    else: #train a specific classifier for this EC class
        df_train = df_train[(df_train.mainclass_set==ec_class)]
        if(train_val_ratio<1):
            df_test["cluster_ID"]=2
        df_train.ecs = df_train.ecs.apply(lambda x: ["NoEC"] if (len(x)==0 or int(x[0][0])!=ec_class) else x)#mark all other classes as NoEC
        df_test.ecs = df_test.ecs.apply(lambda x: ["NoEC"] if (len(x)==0 or int(x[0][0])!=ec_class) else x)#mark all other classes as NoEC

    df_ec = pd.concat([df_train,df_test])
    return df_ec

def eval_ecpred(y_true123456, y_pred123456, label_itos=["1","2","3","4","5","6","NoEC"], level=1):
    '''evaluates ec predictions in a fashion consistent with the ECPred paper
    y_true12345: list of 6 np.arrays each of them of shape [samples_testsetx] on for each EC test set
    y_pred12345: list of 6 np.arrays each of them of shape [samples_testsetx, classes] (classes=2 for level=0 and classes=7 for level=1)
    label_itos: mapping numerical indices (in y_true/y_pred) to labels i.e. EC numbers; normally ["EC","NoEC"] for level 0 and ["1","2","3","4","5","6","NoEC"] for level 1
    level: EC prediction level
    '''
    
    def single_eval_ecpred(y_true,y_pred,selected_label="1",label_itos=["1","2","3","4","5","6","NoEC"]):
        id_selected = np.where(np.array(label_itos)==selected_label)[0][0]
        y_true_eq_id_selected = (y_true == id_selected)
        y_pred_argmax_id_selected = (np.argmax(y_pred,axis=1) == id_selected)
        return f1_score(y_true_eq_id_selected,y_pred_argmax_id_selected)

    scores = []
    for i in range(6):
        scores.append(single_eval_ecpred(y_true123456[i], y_pred123456[i],selected_label=("EC" if level==0 else str(i+1)), label_itos=label_itos))
    print("Individual Scores (F1):",scores)
    print("Mean score (F1):",np.mean(scores))
    return np.mean(scores)
###########################################################################################################################
#proteins and peptides from fasta
###########################################################################################################################
def parse_uniprot_fasta(fasta_path, max_entries=0):
    '''parse uniprot from fasta file (which contains less information than the corresponding xml but is also much smaller e.g. ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta)'''
    rows=[]
    dataset_dict={"sp":"Swiss-Prot","tr":"TrEMBL"}
    for seq_record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        sid=seq_record.id.split("|")
        accession = sid[1]
        dataset = dataset_dict[sid[0]]
        name = sid[2]
        description = seq_record.description
        sequence=str(seq_record.seq)
        
        #print(description)
        
        m = re.search('PE=\d', description)
        pe=int(m.group(0).split("=")[1])
        m = re.search('OS=.* (?=OX=)', description)
        organism=m.group(0).split("=")[1].strip()
        
        data_dict={"ID": accession, "name": name, "dataset":dataset, "proteinexistence":pe, "organism":organism, "sequence": sequence}
        rows.append(data_dict)
        if(max_entries > 0 and len(rows)==max_entries):
            break
    
    df=pd.DataFrame(rows).set_index("ID")
    df['name'] = df.name.astype(str)
    df['dataset'] = df.dataset.astype('category')
    df['organism'] = df.organism.astype('category')
    df['sequence'] = df.sequence.astype(str)
    return df
    
def proteins_from_fasta(fasta_path):
    '''load proteins (as seqrecords) from fasta (just redirects)'''
    return seqrecords_from_fasta(fasta_path)

def seqrecords_from_fasta(fasta_path):
    '''load seqrecords from fasta file'''
    seqrecords = list(SeqIO.parse(fasta_path, "fasta"))
    return seqrecords

def seqrecords_to_sequences(seqrecords):
    '''converts biopythons seqrecords into a plain list of sequences'''
    return [str(p.seq) for p in seqrecords]

def sequences_to_fasta(sequences, fasta_path, sequence_id_prefix="s"):
    '''save plain list of sequences to fasta'''
    with open(fasta_path, "w") as output_handle:
        for i,s in tqdm(enumerate(sequences)):
            record = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(s), id=sequence_id_prefix+str(i), description="")
            SeqIO.write(record, output_handle, "fasta")

def df_to_fasta(df, fasta_path):
    '''Save column "sequence" from pandas DataFrame to fasta file using the index of the DataFrame as ID. Preserves original IDs in contrast to the function sequences_to_fasta()'''    
    with open(fasta_path, "w") as output_handle:
        for row in df.iterrows():            
            record = Bio.SeqRecord.SeqRecord(Bio.Seq.Seq(row[1]["sequence"]), id=str(row[0]), description="")
            SeqIO.write(record, output_handle, "fasta")

def sequences_to_df(sequences, sequence_id_prefix="s"):
    data = {'ID': [(sequence_id_prefix+str(i) if sequence_id_prefix!="" else i) for i in range(len(sequences))], 'sequence': sequences}
    df=pd.DataFrame.from_dict(data)
    return df.set_index("ID")

def fasta_to_df(fasta_path):
    seqs=SeqIO.parse(fasta_path, "fasta")
    res=[]
    for s in seqs:
        res.append({"ID":s.id,"sequence":str(s.seq)})
    return pd.DataFrame(res)

###########################################################################
# Processing CD-HIT clusters
###########################################################################
def clusters_df_from_sequence_df(df,threshold=[1.0,0.9,0.5],alignment_coverage=[0.0,0.9,0.8],memory=16000, threads=8, exploded=True, verbose=False):
    '''create clusters df from sequence df (using cd hit)
    df: dataframe with sequence information
    threshold: similarity threshold for clustering (pass a list for hierarchical clustering e.g [1.0, 0.9, 0.5])
    alignment_coverage: required minimum coverage of the longer sequence (to mimic uniref https://www.uniprot.org/help/uniref)
    memory: limit available memory
    threads: limit number of threads
    exploded: return exploded view of the dataframe (one row for every member vs. one row for every cluster)

    uses CD-HIT for clustering
    https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide
    copy cd-hit into ~/bin
    
    TODO: extend to psi-cd-hit for thresholds smaller than 0.4
    '''
    
    if verbose:
        print("Exporting original dataframe as fasta...")
    fasta_file = "cdhit.fasta"
    df_original_index = list(df.index) #reindex the dataframe since cdhit can only handle 19 letters
    df = df.reset_index(drop=True)

    df_to_fasta(df, fasta_file)

    if(not(isinstance(threshold, list))):
        threshold=[threshold]
        alignment_coverage=[alignment_coverage]
    assert(len(threshold)==len(alignment_coverage))    
    
    fasta_files=[]
    for i,thr in enumerate(threshold):
        if(thr< 0.4):#use psi-cd-hit here
            print("thresholds lower than 0.4 require psi-cd-hit.pl require psi-cd-hit.pl (building on BLAST) which is currently not supported")
            return pd.DataFrame()
        elif(thr<0.5):
            wl = 2
        elif(thr<0.6):
            wl = 3
        elif(thr<0.7):
            wl = 4
        else:
            wl = 5
        aL = alignment_coverage[i]

        #cd-hit -i nr -o nr80 -c 0.8 -n 5
        #cd-hit -i nr80 -o nr60 -c 0.6 -n 4
        #psi-cd-hit.pl -i nr60 -o nr30 -c 0.3
        if verbose:
            print("Clustering using cd-hit at threshold", thr, "using wordlength", wl, "and alignment coverage", aL, "...")

        fasta_file_new= "cdhit"+str(int(thr*100))+".fasta"
        command = "cd-hit -i "+fasta_file+" -o "+fasta_file_new+" -c "+str(thr)+" -n "+str(wl)+" -aL "+str(aL)+" -M "+str(memory)+" -T "+str(threads)
        if(verbose):
            print(command)
        process= subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        output, error = process.communicate()
        if(verbose):
            print(output)
        if(error !=""):
            print(error)
        fasta_files.append(fasta_file)
        if(i==len(threshold)-1):
            fasta_files.append(fasta_file_new)
        fasta_file= fasta_file_new
        
    #join results from all clustering steps
    if verbose:
        print("Joining results from different clustering steps...")
    for i,f in enumerate(reversed(fasta_files[1:])):
        if verbose:
            print("Processing",f,"...")
        if(i==0):
            df_clusters = parse_cdhit_clstr(f+".clstr",exploded=False)
        else:
            df_clusters2 = parse_cdhit_clstr(f+".clstr",exploded=False)
            for id,row in df_clusters.iterrows():
                members = row['members']
                new_members =  [list(df_clusters2[df_clusters2.repr_accession==y].members)[0] for y in members]
                new_members = [item for sublist in new_members for item in sublist] #flattened
                row['members']=new_members

    df_clusters["members"]=df_clusters["members"].apply(lambda x:[df_original_index[int(y)] for y in x])
    df_clusters["repr_accession"]=df_clusters["repr_accession"].apply(lambda x:df_original_index[int(x)])

    if(exploded):
        return explode_clusters_df(df_clusters)
    return df_clusters

def parse_cdhit_clstr(filename, exploded=True):
    '''Aux. Function (used by clusters_df_from_sequence_df) to parse CD-HITs clstr output file in a similar way as the uniref data
    for the format see https://github.com/weizhongli/cdhit/wiki/3.-User's-Guide#CDHIT

    exploded:  single row for every ID instead of single for every cluster   
        '''
    def save_cluster(rows,members,representative):
        if(len(members)>0):
            rows.append({"entry_id":filename[:-6]+"_"+representative, "members":members, "repr_accession":representative})
            
    rows=[]
    with open(filename, 'r') as f:
        members=[]
        representative=""

        for l in tqdm(f):
            if(l[0]==">"):
                save_cluster(rows,members,representative)
                members=[]
                representative=""
            else:
                member=(l.split(">")[1]).split("...")[0]
                members.append(member)
                if "*" in l:
                    representative = member
    save_cluster(rows,members,representative)
    df=pd.DataFrame(rows).set_index("entry_id")
    
    if(exploded):
        return explode_clusters_df(df)
    return df
###########################################################################
# PSSM FEATURES
###########################################################################

def _sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def _parse_pssm_ascii(path):
    skip_lines = 2
    with open(path) as f:
        src = f.readlines()[skip_lines:]
    cols = [s  for s in src[0].split('\n')[0].split(' ') if s != '']
    pssm = []
    end = 7
    for line in src[1:-end]:
        l = l = np.array(line.split()[2:22]).astype(int)
        pssm.append(l)
    pssm = np.array(pssm)
    return pssm 

def _psi_blast_pssm(input):
    file, path, blast_db_path, evalue, num_iterations, sigmoid, tmp_fastas_path ,tmp_pssms_path  = input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7]
    # create and call command
    blastx_cline = NcbipsiblastCommandline(
        cmd='psiblast', 
        query=path+tmp_fastas_path+file, 
        db=blast_db_path, 
        evalue=evalue, 
        num_iterations=num_iterations,
        outfmt=5,
        num_threads=3,
        out_ascii_pssm=path+tmp_pssms_path+file.split('.')[0])()
    if os.path.isfile(path+tmp_pssms_path+file.split('.')[0]):
        pssm = _parse_pssm_ascii(path+tmp_pssms_path+file.split('.')[0])
    else:
        # TODO: what to do in this case?! e.g. no sequences returned by psiblast due to too low evalue
        return None
    if sigmoid:
        return _sigmoid(pssm)
    else:
        return pssm


def compute_pssm_features(query_folder, db_folder, evalue=.001, num_iterations=3, sigmoid=False, train_db=False, n_processes=16, threading=True, blast_dbs_path=Path("../blast_dbs")):

    if not os.path.isfile('/usr/bin/makeblastdb'):
        print("makeblastdb not found, please install package via apt-get install ncbi-blast+")
        return
    db_identifier = db_folder.split('./')[1]
    #path to the temporary directory for dataframes
    blast_dbs_path.mkdir(exist_ok=True)

    # 1. Setup environment for PSI-BLAST Query
    # turn db_folder into fasta file
    DB_FOLDER = Path(db_folder)
    
    # Load preprocessed data
    tok = np.load(DB_FOLDER/'tok.npy')
    itos = np.load(DB_FOLDER/'tok_itos.npy')
    if train_db:
        # use only redundant training sequences as database
        train_IDs = np.load(DB_FOLDER/'train_IDs.npy')
        tok = tok[train_IDs]
        fasta_db_path = str(blast_dbs_path)+"/"+db_identifier+"_train_only.fasta"
        tmp_fastas_path = '/tmp_fastas_train/'
        tmp_pssms_path = '/tmp_pssms_train/'
        db_identifier = db_folder.split('./')[1]+'_train'
    else:
        # use complete swissprot as database
        fasta_db_path = str(blast_dbs_path)+"/"+db_identifier+".fasta"
        tmp_fastas_path = '/tmp_fastas/'
        tmp_pssms_path = '/tmp_pssms/'
        db_identifier = db_folder.split('./')[1]

    db_sequences = [''.join([itos[i] for i in seq[1:]]) for seq in tok] # skip first token
    sequences_to_fasta(db_sequences, fasta_db_path)
    # create database for NCBI-Blast package
    subprocess.call('/usr/bin/makeblastdb -in ' + fasta_db_path + ' -dbtype prot -out ' + str(blast_dbs_path)+"/"+db_identifier + ' -parse_seqids -title ' + db_identifier, shell=True)

    # 2. Preprocess Query-Sequnces
    QUERY_FOLDER = Path(query_folder)
    TMP_QUERY_FOLDER = Path(query_folder+tmp_fastas_path)
    TMP_QUERY_FOLDER.mkdir(exist_ok=True)
    # Load query sequences and store them as individual fasta files
    query_tok = np.load(QUERY_FOLDER/'tok.npy')
    query_IDs = np.load(QUERY_FOLDER/'ID.npy')
    query_train_IDs = np.load(QUERY_FOLDER/'train_IDs.npy')
    query_test_IDs = np.load(QUERY_FOLDER/'test_IDs.npy')
    query_val_IDs = np.load(QUERY_FOLDER/'val_IDs.npy')
    query_itos = np.load(QUERY_FOLDER/'tok_itos.npy')

    query_sequences = [''.join([query_itos[i] for i in seq[1:]]) for seq in query_tok] # skip first token
    for i, seq_id in tqdm(enumerate(np.concatenate((query_train_IDs, query_test_IDs, query_val_IDs)).astype(int))):
        name = query_IDs[seq_id]
        seq = query_sequences[seq_id]
        record = SeqRecord(Seq(seq, ProteinAlphabet), str(name))
        SeqIO.write(record, query_folder+tmp_fastas_path+name + '.fasta', 'fasta')

    # 3. Compute PSSM for each query sequence
    TMP_PSSM_FOLDER = Path(query_folder+tmp_pssms_path)
    TMP_PSSM_FOLDER.mkdir(exist_ok=True)

    files = os.listdir(str(TMP_QUERY_FOLDER))

    if threading:
        n_files = len(files)
        arguments = itertools.zip_longest(
            files, 
            itertools.repeat(query_folder,n_files), 
            itertools.repeat(str(blast_dbs_path)+"/"+db_identifier,n_files), 
            itertools.repeat(evalue,n_files), 
            itertools.repeat(num_iterations,n_files), 
            itertools.repeat(sigmoid,n_files),
            itertools.repeat(tmp_fastas_path,n_files), 
            itertools.repeat(tmp_pssms_path,n_files))

        pssms = [None]*n_files
        with Pool(processes=n_processes) as p:
            max_ = len(files)
            with tqdm(total=max_) as pbar:
                for i, j in tqdm(enumerate(p.imap(_psi_blast_pssm, arguments))):
                    pssms[i] = j
                    pbar.update()
    else:
        pssms = np.array([_psi_blast_pssm([file, query_folder, str(blast_dbs_path)+"/"+db_identifier, evalue, num_iterations, sigmoid, tmp_fastas_path, tmp_pssms_path]) for file in tqdm(files)])

