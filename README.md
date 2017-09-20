Implements the DRAW paper in PyTorch:
https://arxiv.org/abs/1502.04623

draw.py: implementation without attention <br>
draw_attn.py: implementation with attention

Set batch_size to 100. Modify seq_len to change the number of recurrent iterations.

Samples improve with seq_len. 

The attention implementation is very fresh. For one thing, there are blow up issues possibly arising from the recurrences - the gamma parameter that divides in the writer gives nan for seq_len>3. Changing the decoder RNN to LSTM improves things somewhat.


Work in progress.
