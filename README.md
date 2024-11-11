## Aspect-Based Sentiment Analysis with LSTMs and BERT

This report details two approaches for Aspect-Based Sentiment Analysis (ABSA) using LSTMs and BERT. The first approach tackles task A (aspect term identification) and task B (aspect term polarity classification) separately using different embedding techniques. The second approach employs a multi-task BERT model that simultaneously performs both tasks within a cascaded architecture. 

### Approach 1: Separate Models for Task A & B

This approach utilizes separate LSTM models for each task. The following steps are involved:

1. **Data Preprocessing:**
    - Reads training and development datasets (restaurants and laptops).
    - Splits sentences into words, handling punctuation and contractions.
    - Creates a vocabulary and assigns unique IDs to words.
    - Prepares data for both tasks, encoding words with their corresponding IDs.
    - Pads sequences to a fixed maximum length for uniformity.

2. **Embeddings:**
    - Explores three embedding techniques:
        - **Random Embeddings:** Initializes word embeddings randomly.
        - **GloVe Embeddings:** Loads pre-trained GloVe embeddings, providing contextualized word representations.
        - **Word2Vec Embeddings:** Loads pre-trained Word2Vec embeddings, providing semantic relationships between words.

3. **Model Architecture:**
    - **Task A (Aspect Term Identification):** 
        - Uses a bidirectional LSTM to capture contextual information.
        - Employs a linear layer and a dropout layer for classification.
        - Defines a cross-entropy loss function with an ignore index for padding.
        - Calculates F1 scores using a custom function to evaluate performance.
    - **Task B (Aspect Term Polarity Classification):**
        - Follows a similar LSTM architecture as task A.
        - Adapts the linear layers and output classes based on the polarity labels.
        - Uses a separate cross-entropy loss function and F1 score calculation.

4. **Training:**
    - Trains the models using PyTorch Lightning, enabling efficient training and logging.
    - Optimizes the model using the Adam optimizer with a learning rate of 0.0001.
    - Monitors training progress with TensorBoard, visualizing metrics like loss and F1 scores.

### Approach 2: Multi-task BERT Model with Cascaded Architecture

This approach uses a single BERT model to perform both task A and task B within a cascaded architecture. The steps are as follows:

1. **BERT Embeddings:**
    - Utilizes pre-trained BERT embeddings for powerful contextualized representations.
    - Utilizes `BertTokenizerFast` for efficient tokenization and encoding.

2. **Data Preprocessing:**
    - Extends the preprocessing for task A to create a dataset compatible with BERT.
    - Encodes sentences with BERT's tokenizer, including special tokens like [CLS] and [SEP].
    - Prepares the dataset for both task A and task B, associating labels with the corresponding tokens.

3. **Model Architecture:**
    - **BERT:**
        - Uses a pre-trained BERT model for contextualized word representations.
        - Optionally freezes BERT parameters for feature-based transfer learning.
    - **LSTM for Task A:** 
        - Takes the BERT output as input.
        - Applies an LSTM layer to capture sequential information.
        - Uses a linear layer to project the LSTM output to the number of classes for task A.
    - **LSTM for Task B:**
        - Concatenates the BERT output with the predicted keyword positions from task A.
        - Processes this combined input through another LSTM layer.
        - Uses a linear layer and dropout for task B classification.

4. **Training:**
    - Trains the BERT model with PyTorch Lightning, efficiently handling multi-task learning.
    - Defines separate loss functions for task A and task B.
    - Implements a weighted loss combination to balance the contribution of each task.
    - Monitors training and validation metrics with TensorBoard, tracking loss and F1 scores for both tasks.

### Evaluation and Results

The performance of each approach is evaluated based on the Micro, Macro, and Weighted F1 scores.

| Model Type | Word Embeddings | Task A (F1 Scores) | Task B (F1 Scores) |
|---|---|---|---|
| **Separate Model** | Random Embeddings | micro: 0.952, macro: 0.724, weighted: 0.963 | micro: 0.481, macro: 0.246, weighted: 0.608 |
| | Glove 300 | micro: 0.958, macro: 0.769, weighted: 0.966| micro: 0.607, macro: 0.313, weighted: 0.723 |
| | Word2Vec 300 | micro: 0.955, macro: 0.740, weighted: 0.964 | micro: 0.609, macro: 0.324, weighted: 0.727 |
| **Sequential Model** | BERT | micro: 0.993, macro: 0.856, weighted: 0.994 | micro: 0.730, macro: 0.462, weighted: 0.795 |


### Conclusion

This project explored two approaches for ABSA, demonstrating the effectiveness of both LSTM and BERT models. The multi-task BERT model emerged as a superior approach, leveraging the power of pre-trained embeddings and cascaded architecture for enhanced performance.  Further research could explore different architectures, hyperparameter tuning, and the use of external resources to improve the accuracy and robustness of ABSA systems. 
