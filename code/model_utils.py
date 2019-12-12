#for log header
import datetime
import subprocess

# follow installation instructions for fastai v1
from fastai import *        # Quick accesss to most common functionality
from fastai.text import *   # Quick accesss to NLP functionality
from fastai import version

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_auc_score, f1_score

import matplotlib
######################################################################################################
#AUX. FUNCTIONS
######################################################################################################
def flatten_lists(li):
    if isinstance(li, (list, tuple)):
        for item in li:
            yield from flatten_lists(item)
    else:
        yield li

def append_csvlogger_history(folder,kwargs,history_filename="history.csv",csv_filename="results.csv",csv_export=False):
    with (folder/history_filename).open('r') as f:
        hist = f.read().replace("tensor(","").replace(")","")
        write_log(folder,hist)
        if(csv_export):
            res_valid = (hist.split("\n")[-2]).split(",")#[2:]#skip epoch and train loss
            #print("hist",hist)
            #print("res_valid",res_valid)
            write_csv(folder, kwargs, res_valid, filename="results.csv", append=True)

def validate_log_csv(learn, folder, dl=None, csv_file = None, kwargs=None):
    res_valid = learn.validate(dl=learn.data.valid_dl if dl is None else dl,metrics=learn.metrics)
    res_valid = [(r.numpy().tolist() if isinstance(r,Tensor) else r) for r in res_valid ]
    res_valid = list(flatten_lists(res_valid))
    res_str_valid = "validation result:"+" ".join([str(t) for t in res_valid])
    #print(res_str_valid)
    write_log(folder,res_str_valid)
    if(csv_file is not None):
        write_csv(folder, kwargs, res_valid, filename=csv_file, append=True)
    return res_valid

def write_log_header(path, kwargs, filename="logfile.log",append=True):
    if "self" in kwargs:
        del kwargs["self"]
    
    print("======================================\nCommand:"," ".join(sys.argv))
    time = datetime.datetime.now() 
    print("started at ",time)
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    print("Commit:",commit)
    fastai_version = version.__version__
    print("fastai version:", fastai_version)
    
    
    print("\nArguments:")
    for k in sorted(kwargs.keys()):
        print(k,":",kwargs[k])
    print("")

    filepath=path/filename
    with filepath.open("w" if (append is False or filepath.exists() is False) else "a", encoding ="utf-8") as f:
        f.write("\n\nCommand "+" ".join(sys.argv)+"\n")
        f.write("started at "+str(time)+"\n")
        f.write("Commit "+str(commit)+"\n")
        f.write("fastai version "+str(fastai_version)+"\n")
        f.write("\nArguments:\n")
        for k in sorted(kwargs.keys()):
            f.write(k+": "+str(kwargs[k])+"\n")
        f.write("\n") 

def write_log(path, text, filename="logfile.log",append=True):
    filepath=path/filename
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        f.write(str(text)+"\n")

def write_csv(path, kwargs, metrics, filename="results.csv", append=True):
    filepath=path/filename
    file_exists = filepath.exists()
    with filepath.open("w" if append is False else "a", encoding ="utf-8") as f:
        if(file_exists is False or append is False):
            f.write("#")
            for k in sorted(kwargs.keys()):
                f.write(k+"\t")
            f.write("loss\t")

            if(len(kwargs["metrics"])>0):
                for m in kwargs['metrics']:
                    f.write(m+"\t")
            else:#in case kwargs["metrics"] was not passed correctly
                for i in len(metrics):
                    f.write("metric"+str(i)+"\t")
        f.write("\n")
        for k in sorted(kwargs.keys()):
            f.write(str(kwargs[k])+"\t")
        for m in metrics:
            f.write(str(m).replace("tensor(","").replace(")","")+"\t")

def move_plot(path, filename="loss"):
    '''move plot from working directory into path'''
    src_png=Path("./loss_plot.png")
    dst_png=path/(filename+".png")
    src_npy=Path("./losses.npy")
    dst_npy=path/(filename+".npy")
    
    src_png.rename(dst_png)
    src_npy.rename(dst_npy)

def lr_find_plot(learner, path, filename="lr_find", n_skip=10, n_skip_end=2):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    losses = [ to_np(x) for x in learner.recorder.losses[n_skip:-(n_skip_end+1)]]
    #print(learner.recorder.val_losses)
    #val_losses = [ to_np(x) for x in learner.recorder.val_losses[n_skip:-(n_skip_end+1)]]

    plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],losses )
    #plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],val_losses )

    plt.xscale('log')
    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

def losses_plot(learner, path, filename="losses", last:int=None):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("Batches processed")

    last = ifnone(last,len(learner.recorder.nb_batches))
    l_b = np.sum(learner.recorder.nb_batches[-last:])
    iterations = range_of(learner.recorder.losses)[-l_b:]
    plt.plot(iterations, learner.recorder.losses[-l_b:], label='Train')
    val_iter = learner.recorder.nb_batches[-last:]
    val_iter = np.cumsum(val_iter)+np.sum(learner.recorder.nb_batches[:-last])
    plt.plot(val_iter, learner.recorder.val_losses[-last:], label='Validation')
    plt.legend()

    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

def score_from_logfile(path, filename="logfile.log",line=-2):
    '''reads result from csv logs in logfile'''
    with (path/filename).open("r") as f:
        lines = f.readlines()
        return [float(x) for x in lines[line].split(",")[:-1]]#drop last entry time from new fastai version

# ######################################################################################################
# #FASTAI EXTENSIONS: METRICS
# ######################################################################################################
def mse_flat(preds,targs):
    #print(preds.size(),targs.size())
    #input()
    return torch.mean(torch.pow(preds.view(-1)-targs.view(-1),2))


def crossentropy_hierarchical(preds, targs, hierarchy=[[0,6]]):
    loss = None
    for h in hierarchy:
        l = F.cross_entropy(preds[:,h[0]:h[1]],torch.argmax(targs[:,h[0]:h[1]],dim=1))
        if loss is None:
            loss = l
        else:
            loss += l
    return loss
    
def accuracy_hierarchical(preds,targs,hierarchy=[6]):
    targs = torch.argmax(targs[:,hierarchy[-1]:-1],dim=1)
    preds = torch.argmax(preds[:,hierarchy[-1]:-1],dim=1)
    return (preds == targs).float().mean()

def crossentropy_mask(preds, targs, ignore_idx=None):
    '''crossentropy loss with flattening operation (for annotation) disregarding label specified via ignore_idx'''
    preds_flat = preds.view((-1,preds.size()[-1]))
    targs_flat = targs.view(-1)
    if(ignore_idx is not None):
        selected_indices = (targs_flat!=ignore_idx).nonzero().squeeze()
        return F.cross_entropy(preds_flat[selected_indices],targs_flat[selected_indices])
    return F.cross_entropy(preds_flat,targs_flat)

def accuracy_mask(preds, targs, ignore_idx=None):
    '''accuracy metric with flattening operation (for evaluating annotation)'''
    preds_flat = preds.view((-1,preds.size()[-1]))
    targs_flat = targs.view(-1)
    if(ignore_idx is not None):  
        return accuracy(preds_flat[selected_indices],targs_flat[selected_indices])
    return accuracy(preds_flat,targs_flat)

def spearman_mask(targs, preds, cap=1):
    '''transform qualitative measurements back to interval [0,2] for rank correlation'''  
    minus2 = (targs>=(2*cap)) & (targs<=(3*cap))
    minus4 = (targs>=(4*cap)) & (targs<=(5*cap))
    targs = targs - 2*cap*minus2 - 4*cap*minus4
    return spearmanr(preds, targs)[0]

def kendall_mask(targs, preds, cap=1):
    '''transform qualitative measurements back to interval [0,2] for rank correlation'''  
    minus2 = (targs>=(2*cap)) & (targs<=(3*cap))
    minus4 = (targs>=(4*cap)) & (targs<=(5*cap))
    targs = targs - 2*cap*minus2 - 4*cap*minus4
    return kendalltau(preds, targs)[0]

def f1_regression_mask(targs, preds, cap=1, threshold=0.57437481):
    '''default threshold corresponds to ic50=500'''
    minus2 = (targs>=(2*cap)) & (targs<=(3*cap))
    minus4 = (targs>=(4*cap)) & (targs<=(5*cap))
    targs = targs - 2*cap*minus2 - 4*cap*minus4
    
    binary_targs = targs > threshold
    binary_preds = preds > threshold
    
    return f1_score(binary_targs, binary_preds)    

def roc_auc_regression_mask(targs, preds, cap=1, threshold=0.57437481):
    '''default threshold corresponds to ic50=500'''
    minus2 = (targs>=(2*cap)) & (targs<=(3*cap))
    minus4 = (targs>=(4*cap)) & (targs<=(5*cap))
    targs = targs - 2*cap*minus2 - 4*cap*minus4
    
    binary_targs = targs > threshold
    
    return roc_auc_score(binary_targs, preds)    
  
def one_hot_np(seq, n_classes):
    """aux. function for 1-hot encoding numpy arrays"""
    b = np.zeros((len(seq), n_classes),dtype=int)
    b[np.arange(len(seq)), seq] = 1
    return b

def roc_auc_score50(y_true, y_score):
    lbl,cnt = np.unique(y_true,return_counts=True)

    max_fpr = min(50/cnt[0],1.0)
    res = roc_auc_score(y_true,y_score[:,1],average=None,max_fpr=max_fpr)
    #undo McClich correction
    #mina = 0.5 * max_fpr**2
    #maxa = max_fpr
    #print(mina,maxa)
    #return (2*res-1)*(maxa-mina)+mina
    return res

class metric_func(Callback):
    "Obtains score using user-supplied function func (potentially ignoring targets with ignore_idx)"
    def __init__(self, func, name="metric_func", ignore_idx=None, one_hot_encode_target=True, argmax_pred=False, softmax_pred=True, sigmoid_pred=False,metric_component=None):
        super().__init__()
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name=name

    def on_epoch_begin(self, **kwargs):
        self.y_pred = None
        self.y_true = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        #flatten everything (to make it also work for annotation tasks)
        y_pred_flat = last_output.view((-1,last_output.size()[-1]))
        y_true_flat = last_target.view(-1)

        #optionally take argmax of predictions
        if(self.argmax_pred is True):
            y_pred_flat = y_pred_flat.argmax(dim=1)
        elif(self.softmax_pred is True):
            y_pred_flat = F.softmax(y_pred_flat, dim=1)
        elif(self.sigmoid_pred is True):
            y_pred_flat = torch.sigmoid(y_pred_flat)
        
        #potentially remove ignore_idx entries
        if(self.ignore_idx is not None):
            selected_indices = (y_true_flat!=self.ignore_idx).nonzero().squeeze()
            y_pred_flat = y_pred_flat[selected_indices]
            y_true_flat = y_true_flat[selected_indices]
        
        y_pred_flat = to_np(y_pred_flat)
        y_true_flat = to_np(y_true_flat)

        if(self.one_hot_encode_target is True):
            y_true_flat = one_hot_np(y_true_flat,last_output.size()[-1])

        if(self.y_pred is None):
            self.y_pred = y_pred_flat
            self.y_true = y_true_flat
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred_flat], axis=0)
            self.y_true = np.concatenate([self.y_true, y_true_flat], axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        #access full metric (possibly multiple components) via self.metric_complete
        self.metric_complete = self.func(self.y_true, self.y_pred)
        if(self.metric_component is not None):
            return add_metrics(last_metrics, self.metric_complete[self.metric_component])
        else:
            return add_metrics(last_metrics, self.metric_complete)
