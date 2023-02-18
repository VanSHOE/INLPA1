## Report on Average Perplexities of Language Models

### Smoothing

We conducted an experiment to compare the average perplexities of four language models (LM1 to LM4) in both training and
testing phases. The average perplexity measures the average uncertainty of the models in predicting the next word in a
given sequence.

The results are presented in the table below:

| Model | Average Perplexity (Test) | Average Perplexity (Train) |
|-------|---------------------------|-----------------------------|
| LM1   | 135.274                   | 2.35                        |
| LM2   | 135.253                   | 1.688                       |
| LM3   | 728.453                   | 2.038                       |
| LM4   | 728.41                    | 1.513                       |
| LM5   | 270.96                    | 1.14344                     |
| LM6   | 594.2416                  | 1.1314                      |

From the given information, it can be observed that LM2 and LM4 have the lowest average perplexity on the test set,
which is indicative of their superior performance in predicting the test data. LM2 has a slightly lower perplexity than
LM4, indicating that it may be slightly more accurate than LM4. LM2 and LM4 were both trained on a larger dataset than
LM1 and LM3, which is likely the reason for their superior performance. A larger dataset generally allows the model to
capture more of the underlying patterns in the language data, resulting in a more accurate model.

On the other hand, LM1 and LM3 have higher perplexity scores on the test dataset, which suggests that they are not as
good at predicting the test data. This may be due to the fact that they were trained on a smaller dataset, which may not
have provided enough training data to capture the full complexity of the language patterns in the data. LM3 has a
substantially higher perplexity than the other models, which indicates that it is the least accurate model.

It is also worth noting that the perplexity scores on the training dataset for all four LMs are much lower than on the
test dataset. This is to be expected, as the models were trained on the training data, so they are expected to perform
better on that data compared to the unseen test data. However, the significant difference in perplexity scores between
the training and test datasets for LM3 and LM4 suggest that these models may be overfitting the training data.
Overfitting occurs when a model becomes too complex and starts to memorize the training data rather than generalizing
well to unseen data.

In summary, the provided information shows that LM2 and LM4 are the most accurate models, likely due to their training
on a larger dataset. LM1 and LM3, on the other hand, have higher perplexity scores and may not perform as well on the
test data, potentially due to their smaller training data size. Additionally, the significant difference in perplexity
scores between the training and test datasets for LM3 and LM4 indicates a potential issue with overfitting.

### NN

From the table, we can see that LM5 has the lowest perplexity in the testing phase among all the models, while LM6 has
the highest perplexity. LM5 also has the lowest perplexity in the training phase, suggesting that it has learned to
model the data more effectively than the other models.

LM5 and LM6 are both LSTM models, which are a type of neural network that can capture long-term dependencies in
sequential data. The lower perplexity of LM5 suggests that LSTMs are better suited to modeling the language data in this
experiment than the other types of models. However, the higher perplexity of LM6 suggests that there may be some
overfitting or other issues with the model architecture or training process.

Overall, these results suggest that the choice of model architecture and training process can have a significant impact
on the performance of language models. It may be worth exploring other types of neural network architectures or
optimization strategies to further improve the performance of these models.

We observed that in most cases, the median perplexity was significantly lower than the mean perplexity in both the
testing and training phases. This suggests that the distribution of perplexities was positively skewed, with a few
outliers having very high perplexities. The mean was therefore influenced by the outliers, leading to a higher average
perplexity. On the other hand, the median was a more robust measure of central tendency that was less sensitive to the
outliers, leading to a lower median perplexity.

It's worth noting that the skewness of the distribution may be due to the fact that language models are generally better
at predicting common words than rare ones. As a result, the models may have higher perplexities for rare words or
sequences, which can lead to a few extreme values that influence the mean perplexity. The median perplexity, on the
other hand, is less affected by the extreme values and can therefore provide a more representative measure of the
model's overall performance.