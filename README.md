## To run for Part2:

navigate to script/stage_2_script

#### Run: script_mlp.py

## To run for Part 3:

navigate to script/stage_3_script
#### For ORL Dataset run: script_cnn_ORL.py
#### For MNIST Dataset run: script_cnn_MNIST.py
#### For CIFAR-10 Dataset run: script_cnn_CIFAR.py

## To run for Part 4:
Add all the datasets into the data/stage_4_data
### For Text Classification
add nltk package and follow the below commands on the python console
for stage 4 downloading using nltk
```
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
```
In the stage_4_script side make a directory called .vector_cache and add the glove 100d file into it

then go to stage_4_script and run: script.py

### For Text Generation
install pandas library
run script: script_TG.py

## To run for Part5:

navigate to script/stage_2_script

#### Run: script_cora.py for CORA dataset

#### Run: script_pubmed.py for PUBMED dataset

#### Run: script_citeseer.py for CITESEER dataset
