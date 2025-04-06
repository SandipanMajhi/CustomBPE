# CustomBPE
BPE Tokenizer for LLMs. 

### Training ###
The training was done on Bookcorpus text. The tokenizer part makes the model `uncased`. 

`Unique Vocabulary Size = 150,000`

Check out the `bpe_train.py` for training purposes.

### Versions ###
0.001 : Contains the base tokenizer trained on BookCorpus. 150,000 unique tokens.
<next> : Even larger vocab
<next> : Adaptive on domains

### Tests:
Quite small number of test have been done. Check out at `test.ipynb` file for examples.

### Development
This tokenizer was built mainly for tineeBERT which is my current project. If you find issues. Please submit in `Issues` tab.


