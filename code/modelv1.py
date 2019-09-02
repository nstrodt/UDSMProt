'''code version consistent with version v1 of the fastai library
follow installation instructions at https://github.com/fastai/fastai
'''
import fire
 
import os, sys, shutil

#for fbeta metric
from sklearn.metrics import roc_auc_score, fbeta_score
#import warnings
from torch.nn.modules.loss import MSELoss, L1Loss


from model_utils import *
from fastai.callbacks.csv_logger import CSVLogger
from fastai.callbacks.tracker import EarlyStoppingCallback, SaveModelCallback

kwargs_defaults = {
"cv_fold":-1, #cv-fold -1 for single split else fold
"working_folder":"./lm_sprot", # folder with preprocessed data 
"pretrained_folder":"./lm_sprot", # folder for pretrained model
"model_filename_prefix":"model", # filename for saved model
"model_folder":"",# folder for model to evaluate if train=False (empty string refers to data folder) i.e. model will be located at {model_folder}/models/{model_filename_prefix}_3.pth
"pretrained_model_filename":"model_3_enc", # filename of pretrained model (default for loading a lm encoder); a suffix _enc will load the encoder only otherwise the full model will be loaded

"emb_sz":400, # embedding size
"nh":1150, # number of hidden units
"nl":3, # number of layers

"wd":1e-7, # weight decay
"bptt":70, # backpropagation through time (sequence length fed at a time) in fastai LMs is approximated to a std. deviation around 70, by perturbing the sequence length on a per-batch basis
"max_len":1024, # RNN only- number of tokens for which the loss is backpropagated (last max_len tokens of the sequence) [only for ordinary classification i.e. annotation=False]  see bptt for classification in the paper   BE CAREFUL: HAS TO BE LARGE ENOUGH
"bs":128, # batch size
"dropout":0.5, # dropout rate
"lr":1e-3, # learning rate
"lr_slice_exponent": 2.6, # learning rate decay per layer group (used if interactive_finegrained=False); set to 1 to disable discriminative lrs
"lr_fixed":False, #do not reduce the lr after each finetuning step
"lr_stage_factor": [2,2,5], #if lr_fixed==False reduce lr after each stage by dividing by this factor (allows to pass a list to specify factor for every stage)
"epochs":30, # epochs for (final) tuning step (pass a list with 4 elements to specify all epochs)
"fp16":False, #fp 16 training
"fit_one_cycle": True, #use one-cycle lr scheduling

"backwards": False, #invert order of all
"clip":.25, # gradient clipping
"train":True, # False for inference only

"from_scratch":True, # train from scratch or pretrain
"gradual_unfreezing":True, # unfreeze layer by layer during finetuning


"arch": "AWD_LSTM", # AWD_LSTM, Transformer, TransformerXL, BERT (BERT shares params nh, nl, dropout with the LSTM config)
"nheads":6, # number of BERT/Transformer heads 

"tie_encoder":True, # tie embedding and output for LMs

"max_seq_len":1024, # max. sequence length (required for certain truncation modes and BERT)
"truncation_mode":0, #determines how to deal with long sequences (longer than max_seq_len) 0: keep (default for RNN) 1:keep first max_seq_len tokens (default for BERT) 2:keep last max_seq_len tokens (not consistent with BERT) 3: drop

"eval_on_test":False, #use original test set as validation set (should only be used for train=False or together with concat_train_val)
"eval_on_val_test":False, #use validation set as usual and evaluate on the test set at the end
"concat_train_val":False, #use both train and valid as training set (for fit using all data- has to be used in conjuction with eval_on_test=True)
"metrics":["accuracy"], # array of strings specifying metrics for evaluation (currently supported accuracy, macro_auc, macro_f1, binary_auc, binary_auc50)
"export_preds":False, #outputs validation set predictions as preds.npz (bundled numpy array with entries val_IDs_sorted (validation set IDs corresponding to the entries in IDs.npy), preds (validation set predictions- these are the actual model outputs and might require an additional softmax), targs (the corresponding target labels))

"annotation":False, # for classification only: this is an annotation task (one output per token)
"regression":False, # for classification only: this is a regression task

"early_stopping": "None", #performs early stopping on specified metric (possible entries: valid entries for metrics or trn_loss or val_loss)

"interactive": False, # for execution in juyter environment; allows manual determination of lrs (specifying just the lr for the first finetuning step)
"interactive_finegrained":False # for execution in juyter environment; allows manual determination of lrs (specifying lrs for all finetuning steps)
}

######################################################################################################
#MAIN METHOD
######################################################################################################
def generic_model(clas=True, **kwargs):
    kwargs_in = kwargs.copy()

    for k in kwargs_defaults.keys():
        if(not( k in kwargs.keys()) or kwargs[k] is None):
            kwargs[k]=kwargs_defaults[k]
    
    WORKING_FOLDER = Path(kwargs["working_folder"])
    #adjust default params for BERT/Transformer
    if(kwargs["arch"]=="BERT" or kwargs["arch"]=="Transformer" or kwargs["arch"]=="TransformerXL"):
        if not("dropout" in kwargs_in.keys()):
            kwargs["dropout"]=0.1
        if not("nh" in kwargs_in.keys()):
            kwargs["nh"]=512
        if not("nl" in kwargs_in.keys()):
            kwargs["nl"]=6
        if not("truncation_mode" in kwargs_in.keys()):
            kwargs["truncation_mode"]=1
    
    #set eval_on_test if concat_train_val
    if(kwargs["concat_train_val"] is True):
        if(kwargs["eval_on_test"] is False):
            print("Concat_train_val: Setting eval_on_test to True.")
            kwargs["eval_on_test"] = True
        #if(kwargs["eval_on_val_test"] is True):
        #    print("Concat_train_val: Setting eval_on_val_test to False.")
        #    kwargs["eval_on_val_test"] = False

    
    #convert into list if required
    if(not(isinstance(kwargs["lr_stage_factor"],list))):
        kwargs["lr_stage_factor"]= [kwargs["lr_stage_factor"]]*3
    if(not(isinstance(kwargs["epochs"],list))):
        kwargs["epochs"]= [1,1,2,kwargs["epochs"]]

    kwargs["clas"]=clas
    
    write_log_header(WORKING_FOLDER,kwargs)

    # Load preprocessed data
    tok = np.load(WORKING_FOLDER/'tok.npy', allow_pickle=True)
    if(clas):
        label = np.load(WORKING_FOLDER/'label.npy')
    
    # dtype issue if all sequences of same length (numpy turns array of lists into matrix)
    # turn matrix into array of python lists
    if tok.dtype is np.dtype("int32"):
        tok_list = np.empty(tok.shape[0], dtype=np.object)
        for i in range(tok.shape[0]):
            tok_list[i] = []
            tok_list[i].extend(tok[i].tolist())
        tok = tok_list
        
    #check if multi-label
    if(clas):
        if(kwargs["annotation"] is False):#ordinary classification task
            if(isinstance(label[0],list) or isinstance(label[0],np.ndarray)):
                multi_class = True
            else:
                multi_class = False
        else:#annotation task
            if(isinstance(label[0][0],list) or isinstance(label[0][0],np.ndarray)):
                multi_class = True
            else:
                multi_class = False
    
    #get train/val/test IDs
    assert(kwargs["concat_train_val"] is False or kwargs["eval_on_test"] is True)
    if(kwargs["cv_fold"]==-1):#single split
        train_IDs_raw = np.load(WORKING_FOLDER/'train_IDs.npy',allow_pickle=True)
        val_IDs_raw = np.load(WORKING_FOLDER/'val_IDs.npy',allow_pickle=True)
        test_IDs_raw = np.load(WORKING_FOLDER/'test_IDs.npy',allow_pickle=True)
    else:#CV-fold
        train_IDs_raw = np.load(WORKING_FOLDER/'train_IDs_CV.npy',allow_pickle=True)[kwargs["cv_fold"]]
        val_IDs_raw = np.load(WORKING_FOLDER/'val_IDs_CV.npy',allow_pickle=True)[kwargs["cv_fold"]] 
        test_IDs_raw = np.load(WORKING_FOLDER/'test_IDs_CV.npy',allow_pickle=True)[kwargs["cv_fold"]] 
    train_IDs = train_IDs_raw if kwargs["concat_train_val"] is False else np.concatenate([train_IDs_raw,val_IDs_raw])
    val_IDs = val_IDs_raw if kwargs["eval_on_test"] is False else test_IDs_raw
    if(kwargs["eval_on_val_test"]):
       test_IDs = test_IDs_raw
       if(len(test_IDs)==0):
           assert(kwargs["concat_train_val"] is False), "Avoiding accidental evaluation on validation set"
           write_log(WORKING_FOLDER,"Empty test set: setting eval_on_val_test to False")
           kwargs["eval_on_val_test"]=False

    tok_itos = np.load(WORKING_FOLDER/'tok_itos.npy',allow_pickle=True)
    
    if(kwargs["from_scratch"] is False):#check if the tok_itos lists are consistent and adapt if required
        PRETRAINED_FOLDER = Path(kwargs["pretrained_folder"])
        if((PRETRAINED_FOLDER/'tok_itos.npy').exists()):
            tok_itos_pretrained = np.load(PRETRAINED_FOLDER/'tok_itos.npy')
            #if(len(tok_itos)!=len(tok_itos_pretrained) or np.all(tok_itos_pretrained==tok_itos) is False):#nothing to do
            if(len(tok_itos)!=len(tok_itos_pretrained) or ~np.all(tok_itos_pretrained==tok_itos)):
                assert(len(tok_itos)<=len(tok_itos_pretrained)) #otherwise the vocab size does not work out
                print("tok_itos does not match- remapping...")
                print("tok_itos_pretrained",tok_itos_pretrained)
                print("tok_itos",tok_itos)
                
                write_log(WORKING_FOLDER,"Remapping tok_itos...")
                tok_itos_new = np.concatenate((tok_itos_pretrained,np.setdiff1d(tok_itos,tok_itos_pretrained)),axis=0)
                tok_stoi_new = {s:i for i,s in enumerate(tok_itos_new)}

                tok_itos_map = np.zeros(len(tok_itos),np.int32)
                for i,t in enumerate(tok_itos):
                    tok_itos_map[i]=tok_stoi_new[t]
                np.save(WORKING_FOLDER/'tok_itos.npy',tok_itos_new)
                tok_itos = tok_itos_new
                tok =  np.array([[tok_itos_map[x] for x in t] for t in tok])
                np.save(WORKING_FOLDER/'tok.npy',tok)

    #determine pad_idx
    pad_idx = int(np.where(tok_itos=="_pad_")[0][0])

    if (clas and not(kwargs["regression"])):
        label_itos = np.load(WORKING_FOLDER/'label_itos.npy',allow_pickle=True)
        if(kwargs["from_scratch"] is False and kwargs["pretrained_model_filename"][-4:]!="_enc"):#if trying to load a full model the label_itos has to coincide
            assert(label_itos == np.load(PRETRAINED_FOLDER/'label_itos.npy')), "label_itos of both models have to coincide" #could implement a similar remapping as for tok_itos at some point
    # invert order if desired
    if(kwargs["backwards"] is True):
        for i in range(len(tok)):
            tok[i] = np.flip(tok[i])
            if(clas and kwargs["annotation"]):
                label[i] = np.flip(label[i])

    # truncate toks
    if(kwargs["truncation_mode"]>0):
        truncated_sequences = []
        for i in range(len(tok)):
            if(len(tok[i])>kwargs["max_seq_len"]):
                truncated_sequences.append(i)
                if(kwargs["truncation_mode"] == 2):#keep end
                    tok[i] = tok[i][-kwargs["max_seq_len"]:]
                    if(clas and kwargs["annotation"]):
                        label[i]=label[i][-kwargs["max_seq_len"]:]
                elif(kwargs["truncation_mode"] ==1):#keep start
                    tok[i] = tok[i][:kwargs["max_seq_len"]]
                    if(clas and kwargs["annotation"]):
                        label[i] = label[i][:kwargs["max_seq_len"]]
        if(kwargs["truncation_mode"] ==3):#remove too long sequences from ids (keeping tok)
            train_IDs = np.setdiff1d(train_IDs,truncated_sequences)
            val_IDs = np.setdiff1d(val_IDs,truncated_sequences)
            if(kwargs["eval_on_val_test"]):
                test_IDs = np.setdiff1d(test_IDs,truncated_sequences)
            print("Removed",len(truncated_sequences),"sequences with length longer than",kwargs["max_seq_len"])
        else:
            print("Truncated",len(truncated_sequences),"sequences to length",kwargs["max_seq_len"])
    
    trn_toks = tok[train_IDs]
    val_toks = tok[val_IDs]

    if(clas):
        trn_labels = label[train_IDs]
        val_labels = label[val_IDs]
    
    if(kwargs["eval_on_val_test"]):
        assert(len(test_IDs)>0)
        test_toks = tok[test_IDs]
        if(clas):
            test_labels = label[test_IDs]  
        
    print("number of tokens in vocabulary:",len(tok_itos),"\ntrain/val/total sequences:",len(trn_toks),"/",len(val_toks),"/",len(trn_toks)+len(val_toks))

    itos={i:x for i,x in enumerate(tok_itos)}
    vocab=Vocab(itos)

    ######################
    #prepare databunch and learner
    if(clas is False):#language model
        if(kwargs["arch"]=="BERT"):
            pass
        else:
            #factory method
            #data_lm = TextLMDataBunch.from_ids(path=WORKING_FOLDER, vocab=vocab, train_ids=trn_toks, valid_ids=val_toks, bs=kwargs["bs"])
            #data block API
            src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, path=WORKING_FOLDER, processor=[]), TextList(items=val_toks, vocab=vocab, path=WORKING_FOLDER, processor=[]))
            src = src.label_for_lm()
            data_lm= src.databunch(bs=kwargs["bs"],bptt=kwargs["bptt"])
            
            #set config and arch
            if(kwargs["arch"]== "AWD_LSTM"):
                arch = AWD_LSTM
                config_lm = awd_lstm_lm_config.copy()
                config_lm["emb_sz"]=kwargs["emb_sz"]
                config_lm["n_hid"]=kwargs["nh"]
                config_lm["n_layers"]=kwargs["nl"]
                config_lm["pad_token"]=pad_idx
                config_lm["tie_weights"]=kwargs["tie_encoder"]
                
            elif(kwargs["arch"]=="Transformer" or kwargs["arch"]=="TransformerXL"):
                if(kwargs["arch"]=="Transformer"):
                    arch = Transformer
                    config_lm = tfmer_lm_config.copy()
                else:
                    arch = TransformerXL
                    config_lm = tfmerXL_lm_config.copy()
                config_lm["ctx_len"]=kwargs["max_seq_len"]
                config_lm["d_model"]=kwargs["nh"]
                config_lm["n_layers"]=kwargs["nl"]
                config_lm["n_heads"]=kwargs["nheads"]
                config_lm["tie_weights"]=kwargs["tie_encoder"]

            learn = language_model_learner(data_lm, arch, config=config_lm, pretrained=False, drop_mult=kwargs["dropout"], clip=kwargs["clip"],wd=kwargs["wd"])
        #set metrics for language modelling
        learn.metrics=[]
        for k in kwargs["metrics"]:
            if(k=="accuracy"):
                if(kwargs["arch"]=="BERT"):
                    pass
                else:
                    learn.metrics.append(accuracy)
            else:
                assert False, "Encountered undefined metric:"+str(k)

    else:#classfication

        #set config and arch
        if(kwargs["arch"]== "AWD_LSTM"):
            arch = AWD_LSTM
            config_clas = awd_lstm_clas_config.copy()
            config_clas["emb_sz"]=kwargs["emb_sz"]
            config_clas["n_hid"]=kwargs["nh"]
            config_clas["n_layers"]=kwargs["nl"]
            config_clas["pad_token"]=pad_idx
            
        elif(kwargs["arch"]=="Transformer" or kwargs["arch"]=="TransformerXL"):
            if(kwargs["arch"]=="Transformer"):
                arch = Transformer
                config_clas = tfmer_clas_config.copy()
            else:
                arch = TransformerXL
                config_clas = tfmerXL_clas_config.copy()
            config_clas["ctx_len"]=kwargs["max_seq_len"]
            config_clas["d_model"]=kwargs["nh"]
            config_clas["n_layers"]=kwargs["nl"]
            config_clas["n_heads"]=kwargs["nheads"]
            #config_clas["pad_token"]=pad_idx
            
        if(kwargs["arch"]=="BERT"):
            pass
        elif(kwargs["regression"]):
            # factory method
            #data_clas = TextClasDataBunch.from_ids(path=WORKING_FOLDER, vocab=vocab, train_ids=trn_toks, valid_ids=val_toks,train_lbls=trn_labels,valid_lbls=val_labels, bs=kwargs["bs"], pad_idx=pad_idx) #classes=label_itos, 
            # data block api
            src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, pad_idx=pad_idx,path=WORKING_FOLDER, processor=[]), TextList(items=val_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]))
            src = src.label_from_lists(trn_labels,val_labels, label_cls=FloatList, processor=[])
            data_clas= src.databunch(bs=kwargs["bs"],pad_idx=pad_idx)
            
            if(kwargs["eval_on_val_test"]):
                src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, pad_idx=pad_idx,path=WORKING_FOLDER, processor=[]), TextList(items=test_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]))
                src = src.label_from_lists(trn_labels,test_labels, label_cls=FloatList, processor=[])
                data_clas_test = src.databunch(bs=kwargs["bs"],pad_idx=pad_idx)                

            learn = text_classifier_learner(data_clas, arch, config=config_clas, pretrained=False, bptt=kwargs["bptt"], max_len=kwargs["max_len"], drop_mult=kwargs["dropout"], metrics=[],clip=kwargs["clip"],wd=kwargs["wd"])
            learn.loss_func = mse_flat_inequalities #mse_flat#F.mse_loss #l1_loss   
        else:
            #factory method
            #data_clas = TextClasDataBunch.from_ids(path=WORKING_FOLDER, vocab=vocab, train_ids=trn_toks, valid_ids=val_toks,train_lbls=trn_labels,valid_lbls=val_labels, classes=label_itos, bs=kwargs["bs"], pad_idx=pad_idx, classes=["dummy"])
            #data block api
            src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]), TextList(items=val_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]))
            src = src.label_from_lists(trn_labels,val_labels, classes=label_itos, label_cls=(None if multi_class is False else partial(MultiCategoryList,one_hot=True)), processor=[])
            data_clas= src.databunch(bs=kwargs["bs"],pad_idx=pad_idx)
            
            #debugging GO
            #for xb,yb in data_clas.train_dl:
            #    print("size",yb.size())
            #    print("zeroth",yb[0])
            #    print("all",yb)
            #    print("trn_labels",trn_labels[0])
            #    input()
            
            if(kwargs["eval_on_val_test"]):
                src = ItemLists(WORKING_FOLDER, TextList(items=trn_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]), TextList(items=test_toks, vocab=vocab, pad_idx=pad_idx, path=WORKING_FOLDER, processor=[]))
                src = src.label_from_lists(trn_labels,test_labels, classes=label_itos, label_cls=(None if multi_class is False else partial(MultiCategoryList,one_hot=True)), processor=[])
                data_clas_test= src.databunch(bs=kwargs["bs"],pad_idx=pad_idx)

            learn = text_classifier_learner(data_clas, arch, config=config_clas, pretrained=False, bptt=kwargs["bptt"], max_len=kwargs["max_len"], drop_mult=kwargs["dropout"], metrics=[],clip=kwargs["clip"],wd=kwargs["wd"])
            if(multi_class):
                learn.loss_func = F.binary_cross_entropy_with_logits 
        
        #set metrics for classification
        if(not(kwargs["regression"])):
            for k in kwargs["metrics"]:
                if(k=="accuracy"):
                    if(kwargs["annotation"] is True):
                        learn.metrics.append(partial(accuracy_mask, ignore_idx=ignore_idx))
                    else:
                        learn.metrics.append(accuracy)
                elif(k=="macro_roc_auc" or k=="macro_auc"):
                    macro_roc_auc = metric_func(roc_auc_score, "macro_auc", None if kwargs["annotation"] is False else np.where(label_itos=="_none_")[0][0])
                    learn.metrics.append(macro_roc_auc)
                elif(k=="binary_roc_auc" or k=="binary_auc"):
                    binary_roc_auc = metric_func(partial(roc_auc_score,average=None), "binary_auc", None if kwargs["annotation"] is False else np.where(label_itos=="_none_")[0][0],metric_component=1)
                    learn.metrics.append(binary_roc_auc)
                elif(k=="binary_roc_auc50" or k=="binary_auc50"):
                    #binary_roc_auc50 = metric_func(roc_auc_score50, "binary_auc50", None if kwargs["annotation"] is False else np.where(label_itos=="_none_")[0][0], one_hot_encode_target=False)
                    _,cnt = np.unique(val_labels if kwargs["concat_train_val"] else test_labels,return_counts=True)
                    #print("counts:",cnt)
                    binary_roc_auc50 = metric_func(partial(roc_auc_score,average=None,max_fpr=min(50/cnt[0],1.0)), "binary_auc50", None if kwargs["annotation"] is False else np.where(label_itos=="_none_")[0][0],metric_component=1)
                    learn.metrics.append(binary_roc_auc50)    
                elif(k=="macro_f1"):
                    macro_f1 = metric_func(partial(fbeta_score, beta=1, average='macro'), "macro_f1", None if kwargs["annotation"] is False else np.where(label_itos=="_none_")[0][0],one_hot_encode_target=False, argmax_pred=True)
                    learn.metrics.append(macro_f1)
                else:
                    assert False, "Encountered undefined metric:"+str(k)
        else:#set metrics for regression
            learn.metrics = []
            for k in kwargs["metrics"]:
                if(k=="spearmanr"):
                    spearman = metric_func(spearman_mask, "spearmanr", one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True)
                    learn.metrics.append(spearman)
                elif(k=="kendalltau"):
                    kendall = metric_func(kendall_mask, "kendalltau", one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True)
                    learn.metrics.append(kendall)
                elif(k=="f1_reg"):
                    f1_reg = metric_func(f1_regression_mask, "f1_reg", one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True)
                    learn.metrics.append(f1_reg)
                elif(k=="auc_roc_reg"):
                    auc_roc_reg = metric_func(roc_auc_regression_mask, "auc_roc_reg", one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True)
                    learn.metrics.append(auc_roc_reg)
                else:
                
                    assert False, "Encountered undefined metric:"+str(k)

    #add logger
    learn.callback_fns.append(CSVLogger)

    #add early stopping
    if(kwargs["early_stopping"]!="None"):
        #learn.callback_fns.append(partial(EarlyStoppingCallback, monitor=kwargs["early_stopping"], min_delta=0.01, patience=3))
        learn.callback_fns.append(partial(SaveModelCallback, monitor=kwargs["early_stopping"], every='improvement', name=kwargs["model_filename_prefix"]+'_3'))


    if(kwargs["train"]):
        with open(WORKING_FOLDER/'kwargs.pkl', 'wb') as handle:
            print("Saving kwargs as kwargs.pkl...")
            pickle.dump(kwargs, handle)
            
        if(kwargs["fp16"]):#half-precision training
            learn = learn.to_fp16(clip=0.1)
        if(kwargs["from_scratch"] is False):
            src_model = PRETRAINED_FOLDER/'models'/(kwargs["pretrained_model_filename"]+".pth")
            dest_model = WORKING_FOLDER/'models'/'pretrained.pth'
            dest_model.write_bytes(src_model.read_bytes())
            if(kwargs["pretrained_model_filename"][-4:]=="_enc"):
                print("Loading pretrained encoder from ",kwargs["pretrained_model_filename"])
                learn.load_encoder('pretrained')
            else:
                print("Loading pretrained model from ",kwargs["pretrained_model_filename"])
                learn.load('pretrained')

        if(kwargs["gradual_unfreezing"] is True and kwargs["from_scratch"] is False):
            #train top layer
            print("Unfreezing one layer and finetuning...")
            learn.freeze_to(-1)
            learn.lr_find(1e-5,1e2)
            
            if(kwargs["interactive"] or kwargs["interactive_finegrained"]):
                learn.recorder.plot()#skip_start=0,skip_end=0)
                plt.show()
                lr0=float(input("stage 0 learning rate:"))
            else:
                lr0 = kwargs["lr"]

            lr_string = "stage 0 lr="+str(lr0)
            print(lr_string)
            write_log(WORKING_FOLDER,lr_string)

            lr_find_plot(learn,WORKING_FOLDER,"lr_find_0")
            if(kwargs["epochs"][0]>0):
                if(kwargs["fit_one_cycle"]):    
                    learn.fit_one_cycle(kwargs["epochs"][0], lr0, moms=(0.8,0.7)) #slice(lr0) does not matter here since it is just a single layer
                else:
                    learn.fit_one_cycle(kwargs["epochs"][0], lr0, moms=(0.8,0.7),div_factor=1.)
                    
                append_csvlogger_history(folder=WORKING_FOLDER,kwargs=kwargs)
                losses_plot(learn, WORKING_FOLDER, filename="losses_0")
                validate_log_csv(learn, WORKING_FOLDER)
                #learn.save(kwargs["model_filename_prefix"]+'_0')

            #train top two layers
            print("\nUnfreezing two layers and finetuning...")
            learn.purge()
            learn.freeze_to(-2)
            learn.lr_find()

            lr1_high = lr0 if kwargs["lr_fixed"] else lr0/kwargs["lr_stage_factor"][0] #lr0/2
            if(kwargs["interactive_finegrained"]):
                learn.recorder.plot()#skip_start=0,skip_end=0)
                plt.show()
                lr1_low=float(input("stage 1 learning rate (low; high="+str(lr1_high)+"):"))
            else:
                lr1_low = lr1_high/(kwargs["lr_slice_exponent"]**4)
            
            lr_string= "stage 1 lr=slice("+str(lr1_low)+","+str(lr1_high)+")"
            print(lr_string)
            write_log(WORKING_FOLDER,lr_string)
            #plot into file
            lr_find_plot(learn,WORKING_FOLDER,"lr_find_1")
            
            if(kwargs["epochs"][1]>0):
                if(kwargs["fit_one_cycle"]):
                    learn.fit_one_cycle(kwargs["epochs"][1], slice(lr1_low,lr1_high), moms=(0.8,0.7))
                else:
                    learn.fit_one_cycle(kwargs["epochs"][1], slice(lr1_low,lr1_high), moms=(0.8,0.7),div_factor=1.)
                append_csvlogger_history(folder=WORKING_FOLDER,kwargs=kwargs)
                losses_plot(learn, WORKING_FOLDER, filename="losses_1")
                #validate_log_csv(learn, WORKING_FOLDER)
                #learn.save(kwargs["model_filename_prefix"]+'_1')
            
            #train top three layers
            print("\nUnfreezing three layers and finetuning...")
            learn.purge()
            learn.freeze_to(-3)
            
            learn.lr_find()
            lr2_high = lr1_high if kwargs["lr_fixed"] else lr1_high/kwargs["lr_stage_factor"][1] #lr1_high/2
            if(kwargs["interactive_finegrained"]):
                learn.recorder.plot()#skip_start=0,skip_end=0)
                plt.show()
                lr2_low=float(input("stage 2 learning rate (low; high="+str(lr2_high)+"):"))
            else:
                lr2_low =  lr2_high/(kwargs["lr_slice_exponent"]**4)
            lr_string= "stage 2 lr=slice("+str(lr2_low)+","+str(lr2_high)+")"
            print(lr_string)
            write_log(WORKING_FOLDER,lr_string)
            #plot into file
            lr_find_plot(learn,WORKING_FOLDER,"lr_find_2")
            
            if(kwargs["epochs"][2]>0):
                if(kwargs["fit_one_cycle"]):
                    learn.fit_one_cycle(kwargs["epochs"][2], slice(lr2_low,lr2_high), moms=(0.8,0.7))
                else:
                    learn.fit_one_cycle(kwargs["epochs"][2], slice(lr2_low,lr2_high), moms=(0.8,0.7), div_factor=1.)
                append_csvlogger_history(folder=WORKING_FOLDER,kwargs=kwargs)
                losses_plot(learn, WORKING_FOLDER, filename="losses_2")
                #validate_log_csv(learn, WORKING_FOLDER)
                #learn.save(kwargs["model_filename_prefix"]+'_2')

        #train all layers
        print("\nFinetuning all layers...")
        learn.purge()
        learn.unfreeze()
        learn.lr_find() #always produce lr find plot

        if(kwargs["interactive_finegrained"] is True or ((kwargs["from_scratch"] is True or kwargs["gradual_unfreezing"] is False) and kwargs["interactive"])):#interactive
            learn.recorder.plot()#skip_start=0,skip_end=0)
            plt.show()
            if(kwargs["from_scratch"] or kwargs["gradual_unfreezing"] is False):
                lr3_high = float(input("stage 3 learning rate:"))
                lr3_low = lr3_high #lr3_high/(kwargs["lr_slice_exponent"]**(len(learn.layer_groups)))
            else:
                lr3_high = lr2_high if kwargs["lr_fixed"] else lr2_high/kwargs["lr_stage_factor"][2] #lr2_high/5
                lr3_low=float(input("stage 3 learning rate (low; high="+str(lr3_high)+"):"))
        else:#non-interactive
            if(kwargs["from_scratch"] is True or kwargs["gradual_unfreezing"] is False):
                lr3_high = kwargs["lr"]
            else:
                lr3_high = lr2_high if kwargs["lr_fixed"] else lr2_high/kwargs["lr_stage_factor"][2] #lr2_high/5
            lr3_low = lr3_high/(kwargs["lr_slice_exponent"]**(len(learn.layer_groups)))
        
        lr_string = "stage 3 lr=slice("+str(lr3_low)+","+str(lr3_high)+")"
        print(lr_string)
        write_log(WORKING_FOLDER,lr_string)

        #plot into file
        lr_find_plot(learn,WORKING_FOLDER,"lr_find_3")
        if(kwargs["epochs"][3]>0):
            if(kwargs["fit_one_cycle"]):
                learn.fit_one_cycle(kwargs["epochs"][3], slice(lr3_low,lr3_high),  moms=(0.8,0.7))
            else:
                learn.fit_one_cycle(kwargs["epochs"][3], slice(lr3_low,lr3_high),  moms=(0.8,0.7),div_factor=1.)
                
            append_csvlogger_history(folder=WORKING_FOLDER,kwargs=kwargs)
            losses_plot(learn, WORKING_FOLDER, filename="losses_3")
            
            learn.save(kwargs["model_filename_prefix"]+'_3')
            learn.save_encoder(kwargs["model_filename_prefix"]+'_3'+'_enc')
    else:
        if(kwargs["model_folder"]!=""):
            learn.path = Path(kwargs["model_folder"])
        learn.load(kwargs["model_filename_prefix"]+'_3')

    #always run validate
    result = validate_log_csv(learn, WORKING_FOLDER, kwargs=kwargs)
    
    
    if(kwargs["export_preds"] is True):
        filename_output = "preds_valid.npz" if kwargs["cv_fold"]==-1 else ("preds_valid_fold"+str(kwargs["cv_fold"])+".npz")
        print("Exporting predictions as ",filename_output)
        val_clas_len = [len(x) for x in val_toks]
        val_toks_sorted = sorted(range_of(val_toks), key=lambda t: val_clas_len[t])
        val_IDs_sorted = [val_IDs[x] for x in val_toks_sorted]
        #val_IDs_sorted_full = np.load(CLAS_FOLDER/'ID.npy')[val_IDs_sorted]
        preds, targs = learn.get_preds(ds_type=DatasetType.Valid)
        np.savez(WORKING_FOLDER/filename_output,IDs=val_IDs_sorted,preds=preds,targs=targs)
       
        if(kwargs["eval_on_val_test"] is True):
            filename_output = "preds_test.npz" if kwargs["cv_fold"]==-1 else ("preds_test_fold"+str(kwargs["cv_fold"])+".npz")
            print("Exporting predictions as ",filename_output)
            learn.data = data_clas_test
            val_toks = tok[test_IDs] #use test ID s here
            val_clas_len = [len(x) for x in val_toks]
            val_toks_sorted = sorted(range_of(val_toks), key=lambda t: val_clas_len[t])
            val_IDs_sorted = [test_IDs[x] for x in val_toks_sorted]
            
            preds, targs = learn.get_preds(ds_type=DatasetType.Valid)
            np.savez(WORKING_FOLDER/filename_output,IDs=val_IDs_sorted,preds=preds,targs=targs)
            
    
    if(kwargs["eval_on_val_test"]):
        result_test = validate_log_csv(learn, WORKING_FOLDER, dl=data_clas_test.valid_dl, kwargs=kwargs)
        result = [list(result),list(result_test)]

    #delete tmp model
    tmp_model = WORKING_FOLDER/"models"/"tmp.pth"
    if(tmp_model.exists()):
        tmp_model.unlink()
    #free memory
    learn.destroy()
    
    #save result to file
    filename_output = "result.npy" if kwargs["cv_fold"]==-1 else "result_fold"+str(kwargs["cv_fold"])+".npy"
    np.save(WORKING_FOLDER/filename_output,result)
    
    return result


######################################################################################################
#MAIN CLASS (LEGACY)
######################################################################################################
class Model(object):
    def generic_model(self, **kwargs):
        return generic_model(**kwargs)

    def languagemodel(self, **kwargs):
        return self.generic_model(clas=False, **kwargs)
    
    def classification(self, **kwargs):
        return self.generic_model(clas=True, **kwargs)
    
############################################################
#MAIN with fire
############################################################
if __name__ == '__main__':
    fire.Fire(Model)
