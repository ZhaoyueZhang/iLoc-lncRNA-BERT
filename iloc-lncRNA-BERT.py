import torch
from transformers import AutoTokenizer, AutoModel

import pandas as pd
import numpy as np
import shutil
import sys
import os
import random

from Bio import SeqIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from skmultilearn.model_selection import IterativeStratification
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from criterion import get_criterion

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df_x, df_y, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.df_x = df_x
        self.df_y = df_y
        self.max_len = max_len

    def __len__(self):
        return len(self.df_x) ## of sequences: 243 for homo_lncRNA_multi6_seq

    def __getitem__(self, index):
        df_x = str(self.df_x[index])
        df_x = "".join(df_x.split())

        inputs = self.tokenizer(#.encode_plus
            df_x,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'features': inputs["input_ids"].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'labels': torch.FloatTensor(self.df_y[index])
        }


class BERTClass(torch.nn.Module):
    def __init__(self,dp1,dp2):
        super(BERTClass, self).__init__()
        self.bert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.sequential = torch.nn.Sequential(torch.nn.Dropout(dp1),
            torch.nn.Linear(768, 64),
            torch.nn.Dropout(dp2),
            torch.nn.Linear(64, 3)
            )    
    
    def forward(self, input_ids, attn_mask, token_type_ids):

        hidden_states = torch.stack(list(self.bert_model(input_ids,
            attention_mask = attn_mask,
            token_type_ids = token_type_ids)[0]),dim=0)
        mean_embeddings = torch.mean(hidden_states, dim=1)
        output_probabilities = self.sequential(mean_embeddings)

        return mean_embeddings, output_probabilities

class EarlyStopping:
    def __init__(self, tolerance=3, min_delta=0.1):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def loss_fn(outputs, targets):
    return get_criterion()(outputs, targets)

def truncate_sequence(sequence, max_length):
    if len(sequence) <= max_length:
        return sequence
    else:
        return sequence[-max_length:]

def create_dataframe(string_list):
    # Create an label DataFrame
    df = pd.DataFrame()

    for string in string_list:
        string_dict = {}

        for i, letter in enumerate(string):
            col_name = f'label{i+1}'
            string_dict[col_name] = letter
        df = pd.concat([df,pd.DataFrame.from_dict(string_dict,orient='index').T], ignore_index=True)

    return df

def getXY(filename,MAX_LEN):

    sequences_test = SeqIO.parse(filename, "fasta")

    X, y = [], []
    for record in sequences_test:
        tempseq = truncate_sequence(record.seq, MAX_LEN)
        output = ''.join(tempseq)
        X.append(output)
        y.append(record.id[:3])

    df = create_dataframe(y)

    col_name = [f'label{i+1}' for i in range(3)]
    for col_n in col_name:
        df[col_n] = df[col_n].astype(str).astype(int)
    df.insert(loc=0, column='sequence', value=X)
    pd.set_option('display.max_rows', df.shape[0]+1)

    X, y = df['sequence'], df[col_name]
    X, y = X.to_numpy(), y.to_numpy()

    return X, y


def getmaxtokenizerlen(X):
    maxlen = 0
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    for each in X:
        templen=tokenizer(each,return_tensors='pt')["input_ids"].shape[1]
        if templen > maxlen:
            maxlen=templen
        #print(templen)
    print('Max length of tokenizer:')
    print(maxlen)
    return maxlen

def get_loader(X, y, tokenlen,BATCH_SIZE):
    # Convert data to tensors and create DataLoader
    dataset_new = CustomDataset(X, y, tokenlen)
    loader_new = torch.utils.data.DataLoader(
        dataset_new,batch_size=BATCH_SIZE,
        shuffle=False,num_workers=0) #shuffle=False 顺序抽样
    return loader_new

def writetofile(test_targets, test_outputs,filename):
    gt=open('%s_target.csv'%filename,'a')
    go=open('%s_output.csv'%filename,'a')

    for eachl in test_targets:
        templine=''
        for eacht in eachl:
            templine += ',%s'%eacht
        gt.write('%s\n'%templine.strip(','))
    gt.close()

    for eachl in test_outputs:
        templine=''
        for eacht in eachl:
            templine += ',%s'%eacht
        go.write('%s\n'%templine.strip(','))
    go.close()

def writetbert(bert_output):
    gt=open('bert_output.csv','a')

    for eachl in bert_output:
        templine=''
        for eacht in eachl:
            templine += ',%s'%eacht
        gt.write('%s\n'%templine.strip(','))
    gt.close()

def train(X, y, k, tokenlen, BATCH_SIZE, LEARNING_RATE, EPOCHS, dp1, dp2, MODEL_SAVE_PATH,schedulerstep):
    foldloss = {}
    foldauc_micro = {}
    foldauc_macro = {}
    fold = 0

    k_fold = IterativeStratification(n_splits=k, order=1)
    for train, test in k_fold.split(X, y):

        fold += 1

        X_train, y_train = X[train], y[train]
        train_loader = get_loader(X_train, y_train, tokenlen, BATCH_SIZE)

        X_test, y_test = X[test], y[test]
        test_loader = get_loader(X_test, y_test, tokenlen, BATCH_SIZE)

        min_test_loss = 15
        min_test_target = min_test_output = []

        model = BERTClass(dp1,dp2)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
        if schedulerstep:
            scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=10, threshold=0.0001, min_lr = 1e-7)

        history = {'test_loss': [], 'test_acc':[], 'test_macro': [], 'test_micro': []}

        early_stopping = EarlyStopping(tolerance=3, min_delta=0.1)

        for epoch in range(1, EPOCHS+1):
            train_loss = test_loss = test_correct = 0

            torch.cuda.empty_cache()

            model.train()
            for batch_idx, data in enumerate(train_loader):
                features = data['features'].to(device, dtype=torch.long)
                mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                labels = data['labels'].to(device, dtype=torch.float)

                useless,outputs = model(features, mask, token_type_ids)
                output_prob = torch.sigmoid(outputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()

            train_loss = train_loss/len(train_loader)

            model.eval()
            with torch.no_grad():
                test_targets = []
                test_outputs = []
                bert_fea = []
                for batch_idx, data in enumerate(test_loader):
                    features = data['features'].to(device, dtype=torch.long)
                    mask = data['attention_mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                    labels = data['labels'].to(device, dtype=torch.float)

                    bert_output,outputs = model(features, mask, token_type_ids)
                    output_prob = torch.sigmoid(outputs)
                    loss = loss_fn(outputs, labels)
                    test_loss = test_loss + loss.item()

                    # Calculate validation accuracy
                    predicted_labels = torch.round(output_prob)#torch.sigmoid(outputs))
                    test_correct += torch.sum(predicted_labels == labels).item()

                    test_targets.extend(labels.cpu().detach().numpy().tolist())
                    test_outputs.extend(output_prob.cpu().detach().numpy().tolist())
                    bert_fea.extend(bert_output.cpu().detach().numpy().tolist())
                # Calculate average losses and accuracies
                test_loss = test_loss / len(test_loader)
                if schedulerstep:
                    scheduler.step(test_loss)
                else:
                    pass
                test_accuracy = test_correct / (len(test_loader.dataset) * labels.shape[1])

                # Calculate ROC scores
                test_targets = np.array(test_targets)
                test_outputs = np.array(test_outputs)
                
                test_roc_macro = roc_auc_score(test_targets, test_outputs, average='macro')
                test_roc_micro = roc_auc_score(test_targets, test_outputs, average='micro')

                # Print training/validation statistics
                print("Fold:{}/{}  Epoch:{}/{} Train Loss:{:.4f} Test Loss:{:.4f} ROC Micro: {:.4f} Accuracy:{:.4f} ROC Macro: {:.4f} LR: {}"
                    .format(fold,k, epoch, EPOCHS, train_loss, test_loss, test_roc_micro, test_accuracy, test_roc_macro, optimizer.param_groups[0]['lr']))
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    min_test_target = test_targets
                    min_test_output = test_outputs
                    best_bert_out = np.array(bert_fea)
                    min_test_auc_macro = test_roc_macro
                    min_test_auc_micro = test_roc_micro

            early_stopping(train_loss, test_loss)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                break
        print('*********validation outputs writing*********')
        writetofile(min_test_target, min_test_output,'train')
        writetbert(best_bert_out)

        foldauc_micro['fold{}'.format(fold)] = min_test_auc_micro
        foldauc_macro['fold{}'.format(fold)] = min_test_auc_macro
        foldloss['fold{}'.format(fold)] = min_test_loss

    print('**************************K fold Train & Evaluation finished, Model saving***************************')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print('Average Performance of {} fold cross validation'.format(k))
    foldauc_micro_avg = sum(foldauc_micro.values())/k
    foldauc_macro_avg = sum(foldauc_macro.values())/k
    foldloss_avg = sum(foldloss.values())/k
    print("Average Loss: {:.4f} \t Macro: {:.4f} \t Micro: {:.4f}".format(foldloss_avg,foldauc_macro_avg,foldauc_micro_avg))
    return foldloss_avg, foldauc_macro_avg, foldauc_micro_avg

def getperformance(test_targets, test_outputs):

    # Calculate ROC scores on the new test set
    test_targets = np.array(test_targets)
    test_outputs = np.array(test_outputs)
    try:
        test_roc_macro = roc_auc_score(test_targets, test_outputs, average='macro')
    except ValueError:
        test_roc_macro = 0
    try:
        test_roc_micro = roc_auc_score(test_targets, test_outputs, average='micro')
    except ValueError:
        test_roc_micro = 0

    # Print evaluation metrics on the new test set
    print("Evaluation on the new test set:")
    print("ROC Macro: {:.4f} ROC Micro: {:.4f}".format(test_roc_macro, test_roc_micro))
    return test_roc_macro,test_roc_micro


def evaluation(X, y, tokenlen, BATCH_SIZE,dp1,dp2,MODEL_SAVE_PATH,filename,wf=False):
    test_loss = 0
    test_targets = []
    test_outputs = []

    test_loader = get_loader(X, y, tokenlen, BATCH_SIZE)

    model = BERTClass(dp1,dp2)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            features = data['features'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)

            useless,outputs = model(features, mask, token_type_ids)
            loss = loss_fn(outputs, labels)
            test_loss = test_loss + loss.item()#test_loss + ((1 / (batch_idx + 1)) * loss.item())

            test_targets.extend(labels.cpu().detach().numpy().tolist())
            test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    test_loss = test_loss / len(test_loader)
    print("test loss:%s"%test_loss)

    test_roc_macro,test_roc_micro = getperformance(test_targets, test_outputs)

    if wf:
        writetofile(test_targets, test_outputs, filename)
        print("Targets and Outputs of test dataset have been write to files.")

    return test_loss, test_roc_macro,test_roc_micro


device = torch.device('cuda:5') if torch.cuda.is_available() else torch.device('cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# hyperparameters
MAX_LEN = 2292
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-5
schedulerstep = True
NUMS_LABELS = 3
OUTPUT_SIZE = 3
k = 10 #fold
seed = 81947
dp1 = 0.1
dp2 = 0.1

seed_everything(seed)


train_filename = "/home/zhangzy/lnc/1216homo_seq.fasta"
filename_homotest = "/home/zhangzy/lnc/lncloc_testing.fasta"
filename_mustest = "/home/zhangzy/lnc/1216mus_seq.fasta"

X_train, y_train = getXY(train_filename,MAX_LEN)
tokenlen = 482#getmaxtokenizerlen(X_train)

X_test, y_test = getXY(filename_homotest,MAX_LEN)
X_mus, y_mus = getXY(filename_mustest,MAX_LEN)

#train
MODEL_SAVE_PATH = "2wayloss_final_dnabert_model.pt"


print(each)

os.system("rm *.csv")
os.system("rm *.pt")

foldloss_avg, foldauc_macro_avg, foldauc_micro_avg = train(X_train, y_train,
    k, tokenlen, BATCH_SIZE, LEARNING_RATE, EPOCHS,
    dp1, dp2, MODEL_SAVE_PATH,schedulerstep)

    # Load the homo test set
test_loss, test_roc_macro,test_roc_micro = evaluation(X_test, y_test,
    tokenlen, BATCH_SIZE, dp1, dp2, MODEL_SAVE_PATH,'test',wf=True)

    # Load the new Mus dataset
mus_loss, mus_roc_macro,mus_roc_micro = evaluation(X_mus, y_mus,
    tokenlen, BATCH_SIZE, dp1, dp2, MODEL_SAVE_PATH,'mus',wf=True)

print(each)
print('Train: %.4f,%.4f,%.4f\nTest: %.4f,%.4f,%.4f\nMus: %.4f,%.4f,%.4f\n'%(
    foldloss_avg, foldauc_macro_avg, foldauc_micro_avg,
    test_loss, test_roc_macro,test_roc_micro,
    mus_loss, mus_roc_macro,mus_roc_micro))

