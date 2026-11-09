from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

lengths = [8192, 8192*2, 8192*4, 8192*8, 8192*16]
fft_toep = [19.21, 10.48, 5.12, 2.11, 1.04]
non_fft_toep = [19.08, 6.95, 2.03, np.nan, np.nan]
transformer = [1.75, 1/1.69, 1/4.93, 1/16.9, np.nan]
fa2_transformer = [12.08, 5.06, 1.70, 1/2.12, 1/7.91]
hyena = [11.63, 6.24, 3.09, 1.49, 1/1.29]

fft_toep_tp = [1/x for x in fft_toep]
non_fft_toep_tp = [1/x for x in non_fft_toep]
fa2_transformer_tp = [1/x for x in fa2_transformer]
transformer_tp = [1/x for x in transformer]
hyena_tp = [1/x for x in hyena]
lengths = [l / 1000 for l in lengths]

plt.plot(lengths, fft_toep_tp,)
plt.plot(lengths, fa2_transformer_tp)
plt.plot(lengths, hyena_tp)
# plt.plot(lengths, transformer_tp)
# plt.plot(lengths, non_fft_toep_tp)

plt.scatter(lengths, fft_toep_tp, label='TMM')
plt.scatter(lengths, fa2_transformer_tp, label='Transformer')
plt.scatter(lengths, hyena_tp, label='Hyena')

# plt.xscale('log') 
plt.xlabel('Context Length (k tokens)', size='large')
plt.xticks(size='x-large')
plt.ylabel('Latency (s)', size='large')
plt.yticks(size='x-large')
# plt.yscale('log')
plt.legend(fontsize='large')
plt.show()
plt.close()


# df = pd.DataFrame()
# df['NI Loss'] = ni_loss
# df['CLM Loss'] = clm_loss
# sns.regplot(df, x='CLM Loss', y='NI Loss', ci=None)
# plt.ylabel('Invertibility Loss', size='large')
# plt.xlabel('Causal Loss', size='large')
# plt.show()
# plt.close()

# model = LinearRegression()
# slope, intercept, r_value, p_value, std_err = stats.linregress(clm_loss, ni_loss)
# print (slope, intercept, r_value**2, p_value, std_err)
