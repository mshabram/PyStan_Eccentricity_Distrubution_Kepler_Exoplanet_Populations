import numpy
import pystan
import matplotlib.pyplot as plt
import cProfile
import re

infilex2 = 'data/hk_R2_best_13.dat'

fx2 = open(infilex2,'r')

datay12 = fx2.readline()  ## NEW
firstlinex2 = datay12.split('\n', 1)[0].split(' ')
firstlinex2 = [w.replace('[', '') for w in firstlinex2]
firstlinex2 = [w.replace(']', '') for w in firstlinex2]
firstlinex2 = [w.replace('.', '_') for w in firstlinex2]
fx2.close()

print firstlinex2
datay2 = numpy.genfromtxt(infilex2, dtype=None, skip_header=1,names=firstlinex2 )
title2 = infilex2.split('/')[-1].split('.')[-2]
print title2

#y = numpy.hstack((datay2["H_OBS"],datay2["K_OBS"]))
#print y.shape
#print y
#plt.hist(y)

eccmodel = """
data {
    int<lower=1> Nm;
    int<lower=1> Ndata;
    real<lower=-1,upper=1> hhat[Ndata];
    real<lower=-1,upper=1> khat[Ndata];
    real<lower=0,upper=1> hhat_sigma[Ndata];
    real<lower=0,upper=1> khat_sigma[Ndata];
}
parameters {
    simplex[Nm] f;
    real<lower=0> e_sigma[Nm];
    real<lower=-1,upper=1> h[Ndata];
    real<lower=-1,upper=1> k[Ndata];
}
model {
    real ps_h[Nm];
    real ps_k[Nm];
    e_sigma ~ uniform(0, 0.5);
    for (n in 1:Ndata)
      for (j in 1:Nm) {
        ps_h[j] <- log(f[j]) + normal_log(h[n],0.0,e_sigma[j]);
      }
      increment_log_prob(log_sum_exp(ps_h));
    for (n in 1:Ndata)
      for (j in 1:Nm) {
        ps_k[j] <- log(f[j]) + normal_log(k[n],0.0,e_sigma[j]);
      }
      increment_log_prob(log_sum_exp(ps_k));
    for (n in 1:Ndata)
      hhat[n] ~ normal(h[n], hhat_sigma[n]);
    for (n in 1:Ndata)
      khat[n] ~ normal(k[n], khat_sigma[n]);
}
"""


#parameters {
#    real y;
#}
#model {
#    increment_log_prob(log_sum_exp(log(0.3) + normal_log(y,0,.3), log(0.7) + normal_log(y,0,0.05)));
#}
#



# Jags version
#eccmodel <- function() {
#
#    #Population parameters
#    for (j in 1:Nm) {
#        e.sigma[j] ~ dunif(0, 1)
#        e.phi[j] <- 1/(e.sigma[j]*e.sigma[j])
#        #f[j] <- 1.0/Nm #Eventually update here
#        a[j] <- 1;
#    }
#    #a <- rep(1,Nm)
#    #a <- matrix(1,Nm,1)
#    f ~ ddirch(a[])
#    
#    for (n in 1:Ndata){
#        #True planet properties
#        c[n] ~ dcat(f[]) #Need to check
#        h[n] ~ dnorm(0, e.phi[c[n]]) %_% T(-1,1) #Eventually do multivariate truncated normal
#        k[n] ~ dnorm(0, e.phi[c[n]]) %_% T(-sqrt(1-h[n]*h[n]),sqrt(1-h[n]*h[n]))
#        
#        #Observed planet properties
#        hhat[n] ~ dnorm(h[n], 1.0/(hhat.sigma[n]*hhat.sigma[n])) %_% T(-1,1)
#        khat[n] ~ dnorm(k[n], 1.0/(khat.sigma[n]*khat.sigma[n]))
#        #Note: some of input e-distribution has hhat^2 + khat^2 >1, JAGS complains about initialization when I try to constrain khat^2+hhat^2<1
#    }
#
#}


ecc_dat = {'Nm': 2, 'Ndata': len(datay2["H_OBS"]), 'alpha': [1], 'hhat': datay2["H_OBS"], 'khat': datay2["K_OBS"], 'hhat_sigma': datay2["H_SIGMA"], 'khat_sigma': datay2["K_SIGMA"]}

fit = pystan.stan(model_code=eccmodel, data=ecc_dat,
                  iter=1000, chains=5)

la = fit.extract(permuted=True)  # return a dictionary of arrays
e_sigma_1 = la['e_sigma']
f_1 = la['f']

a = fit.extract(permuted=False)
#print(a)
print(fit)

print(e_sigma_1)

#plt.hist(e_sigma_1)
fit.plot()

#
#fit = pystan.stan(model_code=eccmodel, iter=100000, chains=5)
#
#la = fit.extract(permuted=True)  # return a dictionary of arrays
#y_out = la['y']
#
#a = fit.extract(permuted=False)
#
#print(fit)
#
#print(y_out.shape)
#
#plt.hist(y_out)
#fit.plot()

plt.show()

with open('/Users/meganshabram/Box Sync/stan/ecc_dist_testing/output/'+title2+'_stan_posterior_samples_iter10000_chains5.txt', 'wb') as f2:
    f2.write(b'"e_sigma_1" "f_1"\n')
        
    numpy.savetxt( f2, numpy.hstack(( e_sigma_1.reshape(-1,1), f_1.reshape(-1,1) )))

cProfile.run('re.compile("foo|bar")')


