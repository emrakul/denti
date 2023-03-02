# Test case

Goal: Achieve sufficient accuracy of STT transcribation on a simple case. Let's say it's a dentist observation of a patient according to a specific grammar (described in grammar.g). Main challenge is to correctly transcribe unfamiliar or rare words (plaque, bleeding, dentist argot etc.)

Plan: scrape a few videos from youtube thematically relevant to the present case. Extract words using forced alignment. Train basic classifier on top of marblenet embeddings. Train a basic LM on a randomly generated corpus according to given grammar.

## Notebooks
`main.ipynb` - baseline solution: QuartzNet + trigram-LM + spelling correction

`split.ipynb` - generate train samples for classifier. Download YT videos, extract pronunciation of special words. Use word alignment as shown here [https://pytorch.org/audio/main/tutorials/forced_alignment_tutorial.html]

`classification.ipynb` - use embeddings from MarbleNet and SVM classifier to classify commands.

`lang_model.ipynb` - use KenLM to produce an ARPA language model 

`text_gen.ipynb` - generate sample phrases to be used in language modelling

## Data

`./train_samples` - extracted samples corresponding to specific words

`./generated` - use TTS to produce samples
