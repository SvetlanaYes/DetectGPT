# DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

This project implements the approach described in the article "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature" before official code release. It aims to detect machine-generated text by analyzing the probability distribution of predicted tokens.

The project consists of the following three files:

1. `main.py`: This file contains the main logic of the DetectGPT project. It takes command-line arguments, reads a dataset file, generates perturbations of the input text, computes scores based on language models, and writes the results to a CSV file.

2. `generate_perturbations.py`: This file provides functions for generating perturbations of the input text. It includes methods to replace certain words with a masking pattern, allowing for analysis of the language model's behavior.

3. `compute_scores.py`: This file contains functions for computing scores based on the generated texts. It calculates the log-likelihood scores and uses them to determine the probability curvature of the texts.

## Usage

To use the DetectGPT project, follow these steps:

1. Make sure you have the required dependencies installed, including the `transformers` library.
2. Download or clone the project files to your local machine.
3. Prepare your dataset file in TSV format, where each line contains an ID, document, and text.
4. Open a terminal or command prompt and navigate to the project directory.
5. Run the following command:

```
python main.py -d <dataset_file> -mf <maskfill_model> -lm <language_model>
```

Replace `<dataset_file>` with the path to your dataset file.
Replace `<maskfill_model>` with the model name for mask filling, such as "DeepPavlov/rubert-base-cased" or "bert-base-multilingual-cased".
Replace `<language_model>` with the language model name, such as "xlm-roberta-large" or "GPT2-small".

6. The script will generate perturbations for each text, compute scores using the language model, and save the results in a CSV file.

Note: The script assumes that you have a CUDA-enabled GPU for faster computation. If you don't have a GPU, remove the `.cuda()` calls in the code to use CPU instead.

## Output

The script will produce two output files:

1. `<maskfill_model>_<name>_.json`: This file contains the generated texts and their perturbations in JSON format. The `<maskfill_model>` and `<name>` are derived from the command-line arguments.

2. `id_doc_label_lm_<language_model>_<maskfill_model>_<name>.tsv`: This file is a CSV file containing the ID, document, and computed scores for each text. The `<language_model>`, `<maskfill_model>`, and `<name>` are derived from the command-line arguments.

## Additional Information
For more details and a deeper understanding of the DetectGPT project, refer to the original article "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature" for a comprehensive explanation of the methodology and techniques used.
