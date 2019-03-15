### Jensen’s Inequality

If $p_1, p_2, ... ,p_n$ are positive numbers which sum to $1$ and $f$ is a `real continuous function` that is `convex`, then$f(\sum_{i=1}^n p_i x_i) \le \sum_{i=1}^n p_i f(x_i)$

If $f$ is concave, then the inequality reverses, giving

$$\begin{align}f(\sum_{i=1}^n p_i x_i) \ge \sum_{i=1}^n p_i f(x_i)\end{align}$$.

The special case of equal $p_i = 1/n$ with the concave function `ln x`  gives

$$\begin{align}\ln (\frac{1}{n} \sum_{i=1}^n x_i) \ge \frac{1}{n} \sum_{i=1}^n \ln x_i\end{align}$$.

which can be exponentiated to give the `arithmetic mean-geometric mean` inequality

$$\begin{align}\frac{x_1+x_2+...+x_n}{n} \ge \sqrt{x_1x_2...x_n}\end{align}$$.