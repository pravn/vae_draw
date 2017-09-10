Implements the DRAW paper in PyTorch:
https://arxiv.org/abs/1502.04623

Currently, it only has the recurrent model built in; no attention yet.
Set batch-size to 100.
Modify seq_len.

We can see that the samples produced improve with seq_len.

Need to take a closer look at the loss function. Also, maybe a recurrent model for the writer, etc. Attention is next.