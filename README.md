# Writing-A-Poem

Here is a project on producing a Tang poem by using LSTM. To start with, we have to get the formula of the update of the parameters of the RNN.

## Differentiate LSTM

First we shall look at the normal update of a hidden state of a normal RNN.
$$
h_t=f(h_{t-1},x_t)
$$
In fact, we could write it out in a more detailed way:
$$
z_t=Uh_{t-1}+Wx_t+b
$$
In the above formula, $z_t$ will be the input of a activation function, $x_t$ is the input of now, and $b$ is a bias term. Based on the knowledge above, we'd look at the long short-term memory model (LSTM). LSTM handles the gradient disappears and explosions. LSTM introduces a new internal state $c_t$ to deliver the information linearly and recurrently. 

$$
\begin{equation} \begin{split} 
c_t &= f_t \odot c_{t-1}+i_t \odot \hat{c}_t  \\
h_t &= o_t \odot tanh(c_t) \\
\hat{c}_t &= tanh(W_cx_t+U_ch_{t-1}+b_c)
\end{split} \end{equation}
$$

where $f_t$, $i_t$, $o_t$ are three gates to control the information flows. $f_t$, the forget gate, controls how much information RNN forgets. $i_t$, the input gate, controls how much information now should be kept. $o_t$, the output gate, controls how much information should be sent to the outside. And these vectors are computed as follows:

$$
\begin{equation} \begin{split} 
i_t &= \sigma(W_ix_t+U_ih_{t-1}+b_i) \\
f_t &= \sigma(W_fx_t+U_fh_{t-1}+b_f) \\
o_t &= \sigma(W_ox_t+U_oh_{t-1}+b_o)
\end{split} \end{equation}
$$

And $\sigma$ means the logistic function. And we now start to compute the differentiates.
$$
\begin{equation} \begin{split} 
\frac{\partial{h_t}}{\partial{f_t}} 
&=  \frac{o_t}{f_t}\odot tanh(c_t)+o_t\odot \frac{\partial tanh(c_t)}{\partial f_t}\\
&=o_t \odot \frac{\partial tanh(c_t)}{\partial c_t} \frac{\partial c_t}{\partial f_t}\\
&=o_t \odot (1-(c_t)^2) \frac{\partial (f_t\odot c_{t-1}+i_t\odot \hat{c}_t)}{\partial f_t}\\
&=o_t \odot (1-(c_t)^2) c_{t-1}
\end{split} \end{equation}
$$
