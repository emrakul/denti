# denti
## Notebooks
`main.ipynb` - baseline solution: QuartzNet + trigram-LM + spelling correction

`split.ipynb` - generate train samples for classifier. Download YT videos, extract pronunciation of special words. Use word alignment as shown here [https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html]

`classification.ipynb` - use embeddings from MarbleNet and SVM classifier to classify commands.

`lang_model.ipynb` - use KenLM to produce an ARPA language model 

`text_gen.ipynb` - generate sample phrases to be used in language modelling

## Data

`./train_samples` - extracted samples corresponding to specific words

`./generated` - use TTS to produce samples
