import time
import generate_random_spd
import scipy.sparse.linalg
import scipy.linalg
import matplotlib.pyplot as plt
import numpy as np

times_thompson = []
times_euclidean = []
times_logeuclid = []
op_number = 130
sample_number = 1

for i in range(op_number):
    euclid_samples = []
    thompson_samples = []
    logeuclid_samples = []
    if i%3 == 0:
        print(i)
    for _ in range(sample_number):
        a = generate_random_spd.single_spd_gen(int((i+3)*10), float(36.0))
        b = generate_random_spd.single_spd_gen(int((i+3)*10), float(36.0))
        start = time.time()
        eig_max = list(scipy.sparse.linalg.eigs(a, M=b, k=1, which='LM')[0])[0]
        eig_min = list(scipy.sparse.linalg.eigs(b, M=a, k=1, which='LM')[0])[0]
        distance = np.log(max(eig_max, eig_min))
        finish_time = time.time() - start
        thompson_samples.append(finish_time)
        start = time.time()
        distance = np.linalg.norm(a-b, ord='fro')
        finish_time = time.time() - start
        euclid_samples.append(finish_time)
        start = time.time()
        log_A = scipy.linalg.logm(a)
        log_B = scipy.linalg.logm(b)
        distance = np.linalg.norm((log_A - log_B), ord='fro') 
        finish_time = time.time() - start
        logeuclid_samples.append(finish_time)
    times_euclidean.append(euclid_samples)
    times_thompson.append(thompson_samples)
    times_logeuclid.append(logeuclid_samples)

times_euclidean = np.array(times_euclidean)
times_logeuclid = np.array(times_logeuclid)
times_thompson = np.array(times_thompson)

mean_euclidean = np.mean(times_euclidean, axis=1)
mean_thompson = np.mean(times_thompson, axis=1)
mean_logeuclid = np.mean(times_logeuclid, axis=1)

plt.plot((np.array(range(op_number))+3)*10, mean_euclidean, label='Euclidean')
plt.plot((np.array(range(op_number))+3)*10, mean_thompson, label='Thompson')
plt.plot((np.array(range(op_number))+3)*10, mean_logeuclid, label='Log Euclidean')
plt.legend()
plt.xlabel('Dimension of SPD')
plt.ylabel('Time')
plt.title('Time Complexity Plot of SPD Distance Metrics')
plt.show()
x = np.log((np.array(range(op_number))+3)*10)
# Euclidean Curve fitting
y_e = np.log(mean_euclidean)
m_e, c_e = np.polyfit(x[50:], y_e[50:], 1)
# Thompson curve fitting
y_t = np.log(mean_thompson)
m_t, c_t = np.polyfit(x[50:], y_t[50:], 1)
y_le = np.log(mean_logeuclid)
m_le, c_le = np.polyfit(x[50:], y_le[50:], 1)
plt.plot(np.exp(np.unique(x)), np.exp(m_e*x + c_e), label=f"Euclidean best fit: {round(m_e, 3)}x+{round(c_e, 3)}")
plt.plot(np.exp(np.unique(x)), np.exp(m_t*x + c_t), label=f"Thompson best fit: {round(m_t, 3)}x+{round(c_t, 3)}")
plt.plot(np.exp(np.unique(x)), np.exp(m_le*x + c_le), label=f"Log-Euclidean best fit: {round(m_le, 3)}x+{round(c_le, 3)}")
# plt.loglog((np.array(range(op_number))+3)*10, (np.array(range(op_number))+3)/1000, label='linear')
# plt.loglog((np.array(range(op_number))+3)*10, (np.array(range(op_number))+3)**2/1000, label='quadratic')
# plt.loglog((np.array(range(op_number))+3)*10, (np.array(range(op_number))+3)**3/1000, label=r"x^3")
plt.loglog((np.array(range(op_number))+3)*10, mean_euclidean, label='Euclidean')
plt.loglog((np.array(range(op_number))+3)*10, mean_thompson, label='Thompson')
plt.loglog((np.array(range(op_number))+3)*10, mean_logeuclid, label='Log Euclidean')
plt.legend()
plt.xlabel('Dimension of SPD')
plt.ylabel('Time')
plt.title('log-log Time Complexity Plot of SPD Distance Metrics')
plt.show()
