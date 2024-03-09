# Japanese ASR Dataset for Evaluation
Following [the previous work on the Japanese ASR](https://arxiv.org/pdf/2312.03668.pdf), we employ two Japanese ASR datasets to evaluate our models.

## JSUT Basic5000
[JSUT Basic5000](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) is a small Japanese ASR dataset 
commonly used for evaluating Japanese ASR model.

## CommonVoice 8.0

```python
from datasets import load_dataset

ds = load_dataset("mozilla-foundation/common_voice_8_0", "ja", use_auth_token=True)

def prepare_dataset(batch):
  """Function to preprocess the dataset with the .map method"""
  transcription = batch["sentence"]
  
  if transcription.startswith('"') and transcription.endswith('"'):
    # we can remove trailing quotation marks as they do not affect the transcription
    transcription = transcription[1:-1]
  
  if transcription[-1] not in [".", "?", "!"]:
    # append a full-stop to sentences that do not end in punctuation
    transcription = transcription + "."
  
  batch["sentence"] = transcription
  
  return batch

ds = ds.map(prepare_dataset, desc="preprocess dataset")

```
