functions{
  real epanechnikov(real x_zero, real x, real h){
    real output;
    real u;
    u = fabs((x - x_zero)/h);
    if (u > 1){
      u = 1;
    }
    else {
      u = u;
    }
    output = 0.75 * (1 - pow(u, 2));
    return fmax(output, 0);
  }
  
  int pos_count(matrix weights, int col_num){ // Counts the positive weights in each column
    int count;
    int nrow;
    nrow = rows(weights);
    count = 0;
    for (i in 1:nrow){
      if (weights[i, col_num] > 0){
        count += 1;
      }
    }
    return(count);
  }
  
  vector keep_y(matrix weights, vector y, int col_num, int count){ // Keeps observations with positive weight
    int nrow;
    vector[count] out;
    int my_row;
    my_row = 1;
    nrow = rows(weights);
    for (i in 1:nrow){
      if (weights[i, col_num] > 0){
        out[my_row] = y[i];
        my_row += 1;
      }
    }
    return(out);
  }
}
data {
  int<lower=1> N;
  vector[N] X;
  int M; 
  vector[N] y;
  real x_zero[M];
  real<lower=0> h;
  real<lower=0> rho[2];
}
transformed data{
  real sigma0 = 1e-3; // These were 1e-3, changed to test case with y=cos(x^1.5) instead of y=cos(x)
  real sigma1 = 1e-3;
  vector[M] mu = rep_vector(0, M);
  matrix[N,M] weights;
  vector[N] y_s[M];
  vector[N] X_s[M];
  vector[N] w_s[M];
  int v_len[M];
  for (n in 1:N)
    for (m in 1:M)
      weights[n,m] = epanechnikov(x_zero[m], X[n], h);
  
  for(m in 1:M){
    v_len[m] = pos_count(weights, m);
    y_s[m] = head(keep_y(weights, y, m, v_len[m]), v_len[m]);
    X_s[m] = head(keep_y(weights, X, m, v_len[m]), v_len[m]) - x_zero[m];
    w_s[m] = head(keep_y(weights, col(weights, m), m, v_len[m]), v_len[m]);
  }
}
parameters {
  real beta_0[M]; // One parameter for each interval with its own kernel
  real beta_1[M]; // They look weird like this but the parser is happy
  // real<lower=0> rho0;
  real<lower=0> alpha0;
  // real<lower=0> sigma0;
  // real<lower=0> rho1;
  real<lower=0> alpha1;
  // real<lower=0> sigma1;
  real<lower=0> sigma_y;
}
model {
  matrix[M, M] L_K0;
  matrix[M, M] L_K1;
  matrix[M, M] K0 = cov_exp_quad(x_zero, alpha0, rho[1]);
  matrix[M, M] K1 = cov_exp_quad(x_zero, alpha1, rho[2]);
  real sq_sigma0 = square(sigma0);
  real sq_sigma1 = square(sigma1);
  
  // diagonal elements
  for (m in 1:M)
    K0[m, m] = K0[m, m] + sq_sigma0;
  L_K0 = cholesky_decompose(K0);
  
  // diagonal elements
  for (m in 1:M)
    K1[m, m] = K1[m, m] + sq_sigma1;
  L_K1 = cholesky_decompose(K1);
  
  
  alpha0 ~ normal(0, 2);
  // sigma0 ~ std_normal();
  
  alpha1 ~ normal(0, 2);
  // sigma1 ~ std_normal();
  
  sigma_y ~ cauchy(0, 1);
  target += multi_normal_cholesky_lpdf(to_vector(beta_0) | mu, L_K0);
  target += multi_normal_cholesky_lpdf(to_vector(beta_1) | mu, L_K1);

  for (m in 1:M){
    for (n in 1:v_len[m]){
      target += (normal_lpdf(y_s[m][n] | beta_0[m] + beta_1[m] * X_s[m][n], sigma_y) * w_s[m][n]);
    }
  }
}

