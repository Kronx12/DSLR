___
___Variables:___
___
$$i = i^{th}\ training\ exemple$$
$$n = Numver\ of\ features$$
$$x^{(i)}=\begin{bmatrix}x^{(i)}_0\\...\\x^{(i)}_n\end{bmatrix}\to(n, 1)$$
$$\theta=\begin{bmatrix}\theta_0 \\ ... \\ \theta_n\end{bmatrix}\to(1, n)$$
$$\theta^T=\begin{bmatrix}\theta_0 & ... & \theta_n\end{bmatrix}\to(1, n)$$

___
___Base functions:___
___
$$g(z) = \frac1{1 + e^{-z}}$$
$$h_\theta(x) = g(\theta^Tx)$$

___
___Cost functions:___
___
$$J(\theta) = - \frac1m \sum^m_{i=1} y^i \log(h_\theta(x^i)) + (1 - y^i) \log(1-h_\theta(x_i))$$
$$\frac\partial{\partial\theta_j}J(\theta) = \frac{1}m\sum_{i=1}^m(h_\theta(x^i)-y^i)x^i_j$$
$$\theta_j:=\theta_j-\alpha\frac{\partial(J(\theta))}{\partial\theta_j}$$

___
___Concepts definition:___
___
$$g(Z)=\frac1{1+e^{-z}}$$
$$Check\ si\ ca\ marche\ e^{-z}$$
$$h_\theta(X)=g(\theta^TX)$$
$$\frac\partial{\partial\theta_j}J(\theta)=\frac1m(h_\theta(X)-Y)\times x_j$$

___
___Vectorization:___
___

$$\theta = \theta-\frac\alpha m X^T(g(X\theta)-y)$$

$$\theta = \begin{bmatrix}\theta_0 \\ \theta_1 \\ \theta_2\end{bmatrix} \to \begin{bmatrix}a \\ b \\ c\end{bmatrix}\to (3, 1)$$

$$x = \begin{bmatrix}x_0 \\ ... \\x_n \end{bmatrix}$$

$$x_n = \begin{bmatrix}astronomy & herbology & -\\x_n^{(1)} & x_n^{(2)} & 1\end{bmatrix}$$
