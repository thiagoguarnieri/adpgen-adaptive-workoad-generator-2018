#https://docs.scipy.org/doc/scipy-0.14.0/reference/stats.html
#https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
#https://docs.scipy.org/doc/scipy/reference/stats.html
#THIS SCRIPT GENERATES SYTHETIC USERS (ITS TOTAL ONTIME, OFFTIME, AVERAGE_BITRATE)
#2018
import math
import numpy as np
from pprint import pprint
import random
import scipy
import scipy.stats
import sys
import csv
import gc

#implements markov flow
#first segment is always lowest quality
#round session duration to be divisible by 5 to get proper number of segments
#==============================================================================#
def cbmg(cluster, duration):
	
	bufferSize = 4;
	
	#distribution generated a wrong ontime. At least 4 segments
	if(duration <= 0):
		return 264, 5280, {264:bufferSize, 396:0, 594:0, 891:0, 1337:0, 2085:0, 3127:0}
	
	length = int(duration)
	
	if(length % 5 != 0):
		segments_qtd = (length // 5) + 1
	else:
		segments_qtd = (length // 5)
	
	#available bitrates in world cup 2018
	qtdS = {264:bufferSize, 396:0, 594:0, 891:0, 1337:0, 2085:0, 3127:0}

	#Probabilities
	MCLUSTER1 = (0.9824,0.0102,0.0018,0.0008,0.0018,0.0017,0.0013,0.0530,0.9419,0.0045,0.0002,0.0002,0.0002,0.0001,0.1325,0.0296,0.8246,0.0061,0.0025,0.0034,0.0012,0.2748,0.0127,0.0116,0.6396,0.0424,0.0141,0.0048,0.2423,0.0064,0.0065,0.0068,0.7132,0.0228,0.0020,0.1434,0.0050,0.0065,0.0038,0.0064,0.8298,0.0051,0.4378,0.0073,0.0118,0.0079,0.0081,0.0100,0.5171)
	MCLUSTER2 = (0.8599,0.0533,0.0425,0.0302,0.0097,0.0029,0.0015,0.0309,0.8701,0.0645,0.0223,0.0084,0.0024,0.0014,0.0238,0.0371,0.8739,0.0484,0.0120,0.0032,0.0016,0.0201,0.0149,0.0277,0.8999,0.0331,0.0036,0.0008,0.0136,0.0118,0.0160,0.0488,0.8840,0.0243,0.0015,0.0147,0.0133,0.0136,0.0256,0.0901,0.8361,0.0065,0.0814,0.1147,0.1080,0.1133,0.1419,0.1576,0.2831)
	MCLUSTER3 = (0.6872,0.1359,0.0350,0.0363,0.0562,0.0307,0.0187,0.0459,0.6423,0.2023,0.0353,0.0463,0.0211,0.0067,0.0298,0.0567,0.6161,0.2052,0.0593,0.0243,0.0085,0.0184,0.0239,0.0443,0.6780,0.2018,0.0257,0.0078,0.0071,0.0090,0.0124,0.0285,0.8865,0.0509,0.0056,0.0049,0.0051,0.0070,0.0102,0.0403,0.9132,0.0193,0.0032,0.0030,0.0048,0.0068,0.0169,0.0246,0.9408)
	MCLUSTER4 = (0.9928,0.0041,0.0007,0.0004,0.0007,0.0005,0.0009,0.0869,0.8983,0.0120,0.0007,0.0007,0.0007,0.0007,0.1234,0.0515,0.7953,0.0113,0.0042,0.0051,0.0091,0.2150,0.0166,0.0224,0.6805,0.0404,0.0174,0.0078,0.0489,0.0031,0.0024,0.0031,0.9347,0.0053,0.0025,0.0687,0.0059,0.0065,0.0051,0.0063,0.9035,0.0040,0.2746,0.0160,0.0242,0.0075,0.0141,0.0069,0.6566)
	MCLUSTER5 = (0.9936,0.0038,0.0015,0.0005,0.0002,0.0002,0.0003,0.1290,0.8614,0.0073,0.0009,0.0006,0.0003,0.0005,0.1843,0.0539,0.6774,0.0131,0.0104,0.0091,0.0518,0.1072,0.0041,0.0051,0.8671,0.0092,0.0050,0.0023,0.1445,0.0093,0.0119,0.0240,0.7367,0.0551,0.0186,0.0508,0.0022,0.0030,0.0034,0.0128,0.9112,0.0167,0.2518,0.0153,0.0812,0.0122,0.0267,0.0365,0.5763)
	MCLUSTER6 = (0.8530,0.1013,0.0207,0.0143,0.0092,0.0010,0.0006,0.0426,0.8398,0.0983,0.0116,0.0067,0.0007,0.0003,0.0220,0.0488,0.8464,0.0721,0.0092,0.0010,0.0006,0.0110,0.0158,0.0325,0.9045,0.0348,0.0009,0.0005,0.0066,0.0088,0.0122,0.0242,0.9461,0.0016,0.0004,0.0652,0.0798,0.1156,0.1301,0.1289,0.4721,0.0083,0.0560,0.0686,0.1895,0.2050,0.1300,0.0103,0.3405)
	MCLUSTER7 = (0.5139,0.0514,0.0293,0.0332,0.0620,0.0763,0.2339,0.0474,0.3784,0.1825,0.0553,0.0859,0.0986,0.1519,0.0284,0.0234,0.4412,0.1400,0.0898,0.0813,0.1960,0.0236,0.0139,0.0293,0.4298,0.2008,0.1011,0.2015,0.0125,0.0080,0.0144,0.0252,0.7096,0.1224,0.1079,0.0073,0.0044,0.0075,0.0097,0.0316,0.8636,0.0759,0.0037,0.0019,0.0049,0.0062,0.0116,0.0129,0.9589)
	MCLUSTER8 = (0.7658,0.0401,0.0460,0.0300,0.0330,0.0287,0.0562,0.0230,0.7933,0.0710,0.0308,0.0266,0.0188,0.0365,0.0194,0.0333,0.7757,0.0685,0.0292,0.0189,0.0550,0.0177,0.0177,0.0364,0.7667,0.0923,0.0288,0.0402,0.0153,0.0124,0.0149,0.0428,0.7746,0.0829,0.0571,0.0133,0.0095,0.0088,0.0173,0.0601,0.7601,0.1308,0.0060,0.0045,0.0049,0.0055,0.0112,0.0246,0.9432)


	curr_markov = []
    
    #selects proper cbmg to be used
	if cluster == 1:
		curr_markov = MCLUSTER1
	elif cluster == 2:
		curr_markov = MCLUSTER2
	elif cluster == 3:
		curr_markov = MCLUSTER3
	elif cluster == 4:
		curr_markov = MCLUSTER4
	elif cluster == 5:
		curr_markov = MCLUSTER5
	elif cluster == 6:
		curr_markov = MCLUSTER6
	elif cluster == 7:
		curr_markov = MCLUSTER7
	elif cluster == 8:
		curr_markov = MCLUSTER8

    #set segment arrays. first segment is buffer
	segments = [0] * (segments_qtd + bufferSize)
	segments[0] = 264
	segments[1] = 264
	segments[2] = 264
	segments[3] = 264
    
    #the default player behavior is to get the first segment at the lowest bitrate
	curr_state = 264

    #markov
	for count in range(segments_qtd):
        #get current bitrate
		segments[count + 4] = curr_state
		qtdS[curr_state] += 1
        #define the next bitrate
		if curr_state == 264:
			curr_state = next_bitrate(curr_markov[0], curr_markov[1], curr_markov[2], curr_markov[3], curr_markov[4], curr_markov[5], curr_markov[6])
		elif curr_state == 396:
			curr_state = next_bitrate(curr_markov[7], curr_markov[8], curr_markov[9], curr_markov[10], curr_markov[11], curr_markov[12], curr_markov[13])
		elif curr_state == 594:
			curr_state = next_bitrate(curr_markov[14], curr_markov[15], curr_markov[16], curr_markov[17], curr_markov[18], curr_markov[19], curr_markov[20])
		elif curr_state == 891:
			curr_state = next_bitrate(curr_markov[21], curr_markov[22], curr_markov[23], curr_markov[24], curr_markov[25], curr_markov[26], curr_markov[27])
		elif curr_state == 1337:
			curr_state = next_bitrate(curr_markov[28], curr_markov[29], curr_markov[30], curr_markov[31], curr_markov[32], curr_markov[33], curr_markov[34])
		elif curr_state == 2085:
			curr_state = next_bitrate(curr_markov[35], curr_markov[36], curr_markov[37], curr_markov[38], curr_markov[39], curr_markov[40], curr_markov[41])
		elif curr_state == 3127:
			curr_state = next_bitrate(curr_markov[42], curr_markov[43], curr_markov[44], curr_markov[45], curr_markov[46], curr_markov[47], curr_markov[48])
        
    #lembrar de multiplicar por 5 (tamanho do segmento) a carga
	return round((sum(segments) / (len(segments) * 1.0)), 2), sum(segments) * 5, qtdS
#==============================================================================#
def read_csv(filename):
	my_arrival = []
	
	with open(filename, 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			my_arrival += row
	
	return my_arrival
#==============================================================================#
#define what is the next bitrate; This is defined according the probabilities
def next_bitrate(p264, p396, p594, p891, p1337, p2085, p3127):
	max_264 = p264
	max_396 = max_264 + p396
	max_594 = max_396 + p594
	max_891 = max_594 + p891
	max_1337 = max_891 + p1337
	max_2085 = max_1337 + p2085
	max_3127 = max_2085 + p3127
    
    #get a value uniformly distributed
	selected = random.uniform(0, 1)

	if(selected <= max_264):
		return 264
	elif(selected > max_264 and selected <= max_396):
		return 396
	elif(selected > max_396 and selected <= max_594):
		return 594
	elif(selected > max_594 and selected <= max_891):
		return 891
	elif(selected > max_891 and selected <= max_1337):
		return 1337
	elif(selected > max_1337 and selected <= max_2085):
		return 2085
	elif(selected > max_2085):
		return 3127
#==============================================================================#
#MAIN
#these two are needed:
np.random.seed(int(sys.argv[2]))
random.seed(int(sys.argv[2]))

#the execution round
exec_round = sys.argv[1]

#seting clusters. Gerando 30%. Depois pegar metade aleaória de cada cluster
CLUSTER1 = 79423 #476542 # 1588474
CLUSTER2 = 237127 #1422764 # 4742548
CLUSTER3 = 330272 #1981636 # 6605453
CLUSTER4 = 57606 #345641 # 1152139
CLUSTER5 = 266125 #1596751 # 5322503
CLUSTER6 = 119469 #716818 # 2389396
CLUSTER7 = 390355 #2342134 # 7807114
CLUSTER8 = 482203 #2893218 # 9644061


clients_list = CLUSTER1 + CLUSTER2 + CLUSTER3 + CLUSTER4 + CLUSTER5 + CLUSTER6 + CLUSTER7 + CLUSTER8
#sys.exit()

other_metrics = []
user_ids_on = []
user_ids_off = []
user_ids_qtd = []
ident = 0

print("EXECUTION ROUND " + str(exec_round));
print("TOTAL USERS REQUESTED " + str(clients_list));
print("VECTORIZED VERSION");

#==============================================================================#
print("CLUSTER 1 - Desktop-Single-LQ");
user_q1 = np.ones(CLUSTER1)
user_on1 = np.floor(scipy.stats.powerlognorm.rvs(3.897280419574708499e-02, 4.113222865153783880e-01, loc = -5.055726964952947711e+00, scale = 1.302260690176909819e+01, size = CLUSTER1, random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on1)):
	while user_on1[count] < 0:
		user_on1[count] = np.floor(scipy.stats.powerlognorm.rvs(3.897280419574708499e-02, 4.113222865153783880e-01, loc = -5.055726964952947711e+00, scale = 1.302260690176909819e+01, size = 1, random_state = int(sys.argv[2])))

for count in range(len(user_on1)):
	avg_bitrate, session_load, qtdSe = cbmg(1, user_on1[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#
print("CLUSTER 2 - Mobile-Mix-VQ");
user_q2 = scipy.stats.nbinom.rvs(0.50, 0.50, loc = 1, size = CLUSTER2, random_state = int(sys.argv[2]))
user_on2 = np.floor(scipy.stats.exponweib.rvs(1.835571784745963608e+00, 4.921706103985519221e-01, loc = -1.359202972715526896e-27, scale = 2.673873759814807158e+02, size=np.sum(user_q2), random_state = int(sys.argv[2])))
user_off2 = np.floor(scipy.stats.exponweib.rvs(1.830776957674923988e+00, 4.926893429914639411e-01, loc = 1.799999999999999716e+02, scale = 4.438503874278322883e+02, size=(np.sum(user_q2) - CLUSTER2), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on2)):
	while user_on2[count] < 0:
		user_on2[count] = np.floor(scipy.stats.exponweib.rvs(1.835571784745963608e+00, 4.921706103985519221e-01, loc = -1.359202972715526896e-27, scale = 2.673873759814807158e+02, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off2)):
	while user_off2[count] < 0:
		user_off2[count]  = np.floor(scipy.stats.exponweib.rvs(1.830776957674923988e+00, 4.926893429914639411e-01, loc = 1.799999999999999716e+02, scale = 4.438503874278322883e+02, size=1, random_state = int(sys.argv[2])))
#----------------------------
for count in range(len(user_on2)):
	avg_bitrate, session_load, qtdSe = cbmg(2, user_on2[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#

print("CLUSTER 3 - Desktop-Mix-MQ");
#pearson 3 equals gengamma
user_q3 = scipy.stats.nbinom.rvs(0.33, 0.38, loc = 1, size = CLUSTER3, random_state = int(sys.argv[2]))
user_on3 = np.floor(scipy.stats.exponpow.rvs(7.553711423906210864e-01, loc = -5.448447451519798048e-24, scale = 6.573489289305207421e+03, size=np.sum(user_q3), random_state = int(sys.argv[2])))
user_off3 = np.floor(scipy.stats.pearson3.rvs(2.391553747244371753e+00, loc = 9.025628374725336016e+02, scale = 8.640239307884819482e+02, size=(np.sum(user_q3) - CLUSTER3), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on3)):
	while user_on3[count] < 0:
		user_on3[count] = np.floor(scipy.stats.exponpow.rvs(7.553711423906210864e-01, loc = -5.448447451519798048e-24, scale = 6.573489289305207421e+03, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off3)):
	while user_off3[count] < 0:
		user_off3[count] = np.floor(scipy.stats.pearson3.rvs(2.391553747244371753e+00, loc = 9.025628374725336016e+02, scale = 8.640239307884819482e+02, size=1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on3)):
	avg_bitrate, session_load, qtdSe = cbmg(3, user_on3[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#

print("CLUSTER 4 - Desktop-Mult-LQ");
user_q4 = scipy.stats.nbinom.rvs(0.61, 0.24, loc = 2, size = CLUSTER4, random_state = int(sys.argv[2]))
user_on4 = np.floor(scipy.stats.lognorm.rvs(1.361331395169104619e+00, loc = -2.647560023213403557e+00, scale = 1.584657369006932868e+02, size=np.sum(user_q4), random_state = int(sys.argv[2])))
user_off4 = np.floor(scipy.stats.weibull_min.rvs(6.424533122463826906e-01, loc = 1.799999999999999716e+02, scale = 8.936772950713620958e+02, size=(np.sum(user_q4) - CLUSTER4), random_state = int(sys.argv[2])))

#avoiding invalid values
for count in range(len(user_on4)):
	while user_on4[count] < 0:
		user_on4[count] = np.floor(scipy.stats.lognorm.rvs(1.361331395169104619e+00, loc = -2.647560023213403557e+00, scale = 1.584657369006932868e+02, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off4)):
	while user_off4[count] < 0:
		user_off4[count] = np.floor(scipy.stats.weibull_min.rvs(6.424533122463826906e-01, loc = 1.799999999999999716e+02, scale = 8.936772950713620958e+02, size= 1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on4)):
	avg_bitrate, session_load, qtdSe = cbmg(4, user_on4[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#
	
print("CLUSTER 5 - Mobile-Mix-LQ");
user_q5 = scipy.stats.nbinom.rvs(0.46, 0.38, loc = 1, size = CLUSTER5, random_state = int(sys.argv[2]))
user_on5 = np.floor(scipy.stats.lognorm.rvs(1.749113063972807636e+00, loc = -1.054558319418806622e+00, scale = 1.642389651082370108e+02, size=np.sum(user_q5), random_state = int(sys.argv[2])))
user_off5 = np.floor(scipy.stats.gengamma.rvs(2.068853381643050326e+00, 4.070356904491857497e-01, loc = 1.799999999999999716e+02, scale = 9.157162897063822982e+01, size=(np.sum(user_q5) - CLUSTER5), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on5)):
	while user_on5[count] < 0:
		user_on5[count] = np.floor(scipy.stats.lognorm.rvs(1.749113063972807636e+00, loc = -1.054558319418806622e+00, scale = 1.642389651082370108e+02, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off5)):
	while user_off5[count] < 0:
		user_off5[count] = np.floor(scipy.stats.gengamma.rvs(2.068853381643050326e+00, 4.070356904491857497e-01, loc = 1.799999999999999716e+02, scale = 9.157162897063822982e+01, size=1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on5)):
	avg_bitrate, session_load, qtdSe = cbmg(5, user_on5[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#

print("CLUSTER 6 - Desktop-Mix-VQ");
user_q6 = scipy.stats.nbinom.rvs(0.33, 0.44, loc = 1, size = CLUSTER6, random_state = int(sys.argv[2]))
user_on6 = np.floor(scipy.stats.beta.rvs(5.377181251700184905e-01, 3.555129443825715718e+00, loc = -5.733985722428243605e-29, scale = 2.126818755729622353e+04, size=np.sum(user_q6), random_state = int(sys.argv[2])))
user_off6 = np.floor(scipy.stats.gengamma.rvs(2.151757185390019700e+00, 4.234648488061530935e-01, loc = 1.799999999999999716e+02, scale = 8.686294856407410236e+01, size=(np.sum(user_q6) - CLUSTER6), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on6)):
	while user_on6[count] < 0:
		user_on6[count] = np.floor(scipy.stats.beta.rvs(5.377181251700184905e-01, 3.555129443825715718e+00, loc = -5.733985722428243605e-29, scale = 2.126818755729622353e+04, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off6)):
	while user_off6[count] < 0:
		user_off6[count] = np.floor(scipy.stats.gengamma.rvs(2.151757185390019700e+00, 4.234648488061530935e-01, loc = 1.799999999999999716e+02, scale = 8.686294856407410236e+01, size=1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on6)):
	avg_bitrate, session_load, qtdSe = cbmg(6, user_on6[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#

print("CLUSTER 7 - Desktop-Mix-HQ");
user_q7 = scipy.stats.nbinom.rvs(0.56, 0.60, loc = 1, size = CLUSTER7, random_state = int(sys.argv[2]))
user_on7 = np.floor(scipy.stats.exponpow.rvs(6.759185064156867373e-01, loc = -1.365055503477551470e-22, scale = 6.115929473743828567e+03, size=np.sum(user_q7), random_state = int(sys.argv[2])))
user_off7 = np.floor(scipy.stats.gamma.rvs(4.443476779111131814e-01, loc = 1.799999999999999432e+02, scale = 1.973961167682356518e+03, size=(np.sum(user_q7) - CLUSTER7), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on7)):
	while user_on7[count] < 0:
		user_on7[count] = np.floor(scipy.stats.exponpow.rvs(6.759185064156867373e-01, loc = -1.365055503477551470e-22, scale = 6.115929473743828567e+03, size=1, random_state = int(sys.argv[2])))
		
for count in range(len(user_off7)):
	while user_off7[count] < 0:
		user_off7[count] = np.floor(scipy.stats.gamma.rvs(4.443476779111131814e-01, loc = 1.799999999999999432e+02, scale = 1.973961167682356518e+03, size=1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on7)):
	avg_bitrate, session_load, qtdSe = cbmg(7, user_on7[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------#
#------------------------------------------------------#
	
print("CLUSTER 8 - Mobile-Mix-HQ");
user_q8 = scipy.stats.nbinom.rvs(0.83, 0.55, loc = 1, size = CLUSTER8, random_state = int(sys.argv[2]))
user_on8 = np.floor(scipy.stats.exponweib.rvs(1.616795762384050983e+00, 5.739057603649651007e-01, loc = -1.402591000762657795e-26, scale = 5.057510039599126230e+02, size=np.sum(user_q8), random_state = int(sys.argv[2])))
user_off8 = np.floor(scipy.stats.gengamma.rvs(1.289453381225011075e+00, 5.630140229748461511e-01,  loc = 1.799999999999999716e+02, scale = 5.901127851224500773e+02, size=(np.sum(user_q8) - CLUSTER8), random_state = int(sys.argv[2])))

#avoiding negative values
for count in range(len(user_on8)):
	while user_on8[count] < 0:
		user_on8[count] = np.floor(scipy.stats.exponweib.rvs(1.616795762384050983e+00, 5.739057603649651007e-01, loc = -1.402591000762657795e-26, scale = 5.057510039599126230e+02, size=1, random_state = int(sys.argv[2])))

for count in range(len(user_off8)):
	while user_off8[count] < 0:
		user_off8[count] = np.floor(scipy.stats.gengamma.rvs(1.289453381225011075e+00, 5.630140229748461511e-01,  loc = 1.799999999999999716e+02, scale = 5.901127851224500773e+02, size=1, random_state = int(sys.argv[2])))
#----------------------------

for count in range(len(user_on8)):
	avg_bitrate, session_load, qtdSe = cbmg(8, user_on8[count])
	other_metrics.append([avg_bitrate, session_load, qtdSe[264], qtdSe[396], qtdSe[594], qtdSe[891], qtdSe[1337], qtdSe[2085], qtdSe[3127]])
#------------------------------------------------------------------------------------------------#
print("CLEANING INDIVIDUAL");
qtd_vals = np.concatenate((user_q1,user_q2,user_q3,user_q4,user_q5,user_q6,user_q7,user_q8))
ontimes_vals = np.concatenate((user_on1,user_on2,user_on3,user_on4,user_on5,user_on6,user_on7,user_on8))
offtimes_vals = np.concatenate((user_off2,user_off3,user_off4,user_off5,user_off6,user_off7,user_off8))

print("IDENTIFICATION");
for count in range(len(qtd_vals)):
	for count2 in range(int(qtd_vals[count])):
		user_ids_on.append([ident,count2])
	
	for count3 in range((int(qtd_vals[count])-1)):
		user_ids_off.append([ident,count3])
		
	user_ids_qtd.append([ident])
	ident = ident + 1

print("Saving sythetic metrics")
np.savetxt('synthetics/cluster_ontime_' + str(exec_round) + '.csv', np.column_stack((user_ids_on,ontimes_vals)), header = "client_id,sess_id,timeon", comments='', delimiter=',', fmt="%1.2f")
np.savetxt('synthetics/cluster_offtime_' + str(exec_round) + '.csv', np.column_stack((user_ids_off,offtimes_vals)), header = "client_id,sess_id,timeoff", comments='', delimiter=',', fmt="%1.2f")
np.savetxt('synthetics/cluster_qtd_' + str(exec_round) + '.csv', np.column_stack((user_ids_qtd,qtd_vals)), header = "client_id,qtd", comments='', delimiter=',', fmt="%1.2f")
np.savetxt('synthetics/cluster_other_' + str(exec_round) + '.csv', np.column_stack((user_ids_on,other_metrics)), header = "client_id,sess_id,avg_bitrate,session_data,q264,q396,q594,q891,q1337,q2085,q3127", comments='', delimiter=',', fmt="%1.2f")
