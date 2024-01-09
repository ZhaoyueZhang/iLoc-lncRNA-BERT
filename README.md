# iLoc-lncRNA-BERT
Data and code of the lncRNA subcellular localization multi-label prediction model iLoc-lncRNA-BERT.

##1. Dataset
  Training set: 219 distinct lncRNA sequences in _H. sapiens_.
  Testing set:  623 distinct lncRNAs from LncLocFormer.
  _M. musculus_ set: 65 distinct lncRNA sequences in _M. musculus_.

##2. Framework
  In the framework of lncRNA subcellular location prediction, as illustrated in Figure 1, we input the truncated lncRNA sequences into the deep learning model. The “DNABERT-2”[1] was used to integrate pre-trained representation of DNA sequences. Subsequently, two linear layers were appended to the “DNABERT-2” architecture, each preceded by a dropout layer for regularization. The first linear layer reduced the dimensionality from 768 to a hidden size of 64. Following this, another dropout layer is employed before the second linear layer, which further reduces the dimensions from 64 to 3. Two-way Multi-Label Loss[2] and Adaptive Moment Estimation(Adam) algorithm were used for the hyperparameter tuning. The Sigmoid activation function was applied to generate the prediction probabilities for each location. 

<img width="416" alt="image" src="https://github.com/ZhaoyueZhang/iLoc-lncRNA-BERT/assets/56220701/6a0f9263-5b9a-4a1e-8667-102d3da20ce2">
Figure 1. The flowchart of multi-label lncRNA subcellular location prediction.

##3. Use iLoc-lncRNA-BERT for lncRNA subcellular location prediction
   ###3.1 Prepare your lncRNA sequences in .fasta format.
   ###3.2 Download 
     1. requirements.txt for Environment Setting
     2. iLoc-lncRNA-BERT_predict.py, criterion.py, and iLoc-lncRNA-BERT.pt for Prediction
   ###3.3 Set Environment
   
   ```# create and activate virtual python environment
   conda create -n dna python=3.8
   conda activate dna

   # (optional if you would like to use flash attention)
   # install triton from source
   git clone https://github.com/openai/triton.git;
   cd triton/python;
   pip install cmake; # build-time dependency
   pip install -e .

   # install required packages
   python3 -m pip install -r requirements.txt```

if "AssertionError occured when run the model(inputs).
    ....flash_attn_triton.py", line 781, in _flash_attn_forward
    assert q.is_cuda and k.is_cuda and v.is_cuda" occurs, 1. Delete the previous environment.2. Build the environment without install triton. 3. Perform "pip uninstall trito"

   ###3.4 Predict
   conda activate dna
   python iLoc-lncRNA-BERT_predict.py example.fasta

   ###3.5 get result
   prediction probabilities be found in result.txt
   prediction probabilities can be found in test_output.csv

##5. Citation
   If you have any question regarding our paper or codes, please feel free to start an issue or email Zhaoyue Zhang (zyzhang@uestc.edu.cn).
   If you use iLoc-lncRNA-BERT in your work, please kindly cite our paper: Zhao-Yue Zhang, Zheng Zhang, Xiucai Ye, Tetsuya Sakurai, Hao Lin. A BERT-based model for the prediction of lncRNA subcellular localization in _Homo sapiens_.

Reference:
[1]	Z. Zhou, Y. Ji, W. Li, P. Dutta, H. Liu. DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome.	arXiv:2306.15006
[2]	T. Kobayashi. Two-way Multi-Label Loss. Proc Cvpr Ieee 2023, 7476-7485.
