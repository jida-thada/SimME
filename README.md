# SimME: Learning Similarity-Preserving Meta-Embedding for Text Mining


## File listing
+ __simme.py__ : Main code for SimME training
+ __simme_utils.py__ : Supporting functions
+ __requirements.txt__ : Library requirements


## Instructions on training SimME

Prepared folders:
+ __work__ : main directory for training 
    + __log__ : directory for log
    + __plot__ : directory for training loss plots
    + __temp_emb__ : check point saving models
+ __sources__ : contains pretrained embedding sources (please paste text file of source embeddings here)
+ __data__ : contains all training word-pairs (please upload your own training word-pairs)
+ __simme_output__ : folder for outputing the SimME meta-embeddings




Run script in work directory as:

### Prepare training data
    python simme.py -embname wikibio --gendata -pairs_filename '../data/pairs.txt' -source_filename '../sources/wiki200.txt' '../sources/biomed.txt' -source_embdim 200 200 
    
### Train SimME meta-embeddings
    python simme.py -embname wikibio --train -embdim 200 -pairs_filename '../data/pairs.txt' -runid 0 
    
    
<b>Parameters:</b>

+ __For prepare training data:__
  + __-embname__ : the embedding name defined by user e.g. wikibio
  + __--gendata__ : boolean parameters, whether to generate training word-pairs with their cosine distance, default false
  + __-pairs_filename__ : the filepath for the training word-pairs uploaded by user in the folder 'data'
  + __-header__ : 'y' if the word-pairs file contains header; 'n' otherwise, default 'n'
  + __-w1colid__ : column index of the 1st word of a word-pair, default 0
  + __-w2colid__ : column index of the 2nd word of a word-pair, default 1
  + __-source_filename__ : the filepaths of all embedding sources 
  + __-source_embdim__ : the dimensions of each embedding source corresponding to the order of such specified paths in source_filename

+ __For training SimME:__
  + __-embname__ : the embedding name defined by user e.g. wikibio
  + __--train__ : boolean parameters, whether to train SimME meta-embeddings, default false
  + __-embdim__ : the target dimension for SimME meta-embeddings defined by user
  + __-pairs_filename__ : a filepath for the training word-pairs uploaded by user in the folder 'data'
  + __-runid__ : running number for experiments
  + __-header__ : 'y' if the word-pairs file contains header; 'n' otherwise, default 'n'
  + __-w1colid__ : column index of the 1st word of a word-pair, default 0
  + __-w2colid__ : column index of the 2nd word of a word-pair, default 1
  + __-lr__ : learning rate, default 15
  + __-ep__ : epochs, default 200
  + __-bs__ : batch size, default 512
  
  
