# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

This moule contain the main function to use to compute the cellular automaton
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from rules0_19 import *
from rules20_39 import *
from rules40_59 import *
from rules60_79 import *
from rules80_99 import *
from rules100_119 import *
from rules120_139 import *
from rules140_159 import *
from rules160_179 import *
from rules180_199 import *
from rules200_219 import *
from rules220_239 import *
from rules240_255 import *
#=============================================================================
def application_rules(shape, rule_num, start='center', seed=None, plot=True,
					 save_path=None, fig_sz1=(10, 10), fig_sz2=(15, 10)):
	"""
	Function to applied one of the 256 possible rules.

	Parameters
	----------
	shape : tuple or int
		Shape/size of the array that will be used for the cellular automata.
	rule_num : int
		The number of the rule that will be used.
	start : str, optional
		Indicate how the the cellular automata will be initialized. The
		default is 'center'.
	seed : NoneType, optional
		Parameter to make its own initialisaion of the automata. It must have
		the same length as the width indicate for the 'shape' parameter. It
		must be a list or numpy.ndarray type. If NoneType, the initialisation
		of the automata will follow the 'start' parameter indication. The
		default is None.
	plot : bool, optional
		Indicate if the result will be ploted. The default is True.
	save_path : str, optional
		Path to where the plot will be saved. The default is None.
	fig_sz1 : tuple, optional
		Size of the figure that will show the final state of the automata.
		The default is (10, 10).
	fig_sz2 : tuple, optional
		Size of the figure that will show the evolution of the automata
		state. The default is (15, 10).

	Raises
	------
	ValueError
		'start' parameter is not from the possible list.
	ValueError
		The asked rule doesn't exist.

	Returns
	-------
	board : numpy.ndarray
		Final state of the automata.
	sum_vert : numpy.ndarray
		Sum of the final state over the axis 0.
	sum_hori : numpy.ndarray
		Sum of the final state over the axis 1.

	Example
	-------
	In [0] : application_rules((13, 13), 15, start='center', seed=None,
							   plot=True, save_path=None, fig_sz1=(10, 10),
							   fig_sz2=(15, 10))

	Out [0]: (np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
						[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
						[0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
						[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
						[1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
						[0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
						[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
						[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
						[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
						[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
						[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]),
			  np.array([5, 4, 8, 7, 6, 5, 9, 5, 6, 7, 8, 4, 5]),
			  np.array([1, 10, 7, 2, 7, 5, 3, 6, 5, 0, 13, 11, 9]))

	"""
	if (type(shape) == int)|(type(shape) == float):
		shape = (int(shape), int(shape))

	board = np.zeros(shape)

	if start == 'center':
		board[0, shape[1]//2] = 1
	elif start == 'bin_w':
		board[0, (np.arange(shape[1])%2) == 0] = 1
	elif start == 'bin_k':
		board[0, (np.arange(shape[1])%2) == 1] = 1
	elif start == 'rand':
		board[0] = np.random.randint(0, 2, shape[1])
	else:
		raise ValueError('start parameter must be from the list: ["center",'
						+ ' "bin_w", "bin_k", "rand"]. Get ' + str(start))

	if type(seed) == np.ndarray:
		board[0] = seed

	list_rules = list(range(256))
	if rule_num not in list_rules:
		raise ValueError("There are 256 rules, from 0 to 255. Get "
						 + str(rule_num))

	if rule_num == 0:
		board = rule0(board)
	elif rule_num == 1:
		board = rule1(board)
	elif rule_num == 2:
		board = rule2(board)
	elif rule_num == 3:
		board = rule3(board)
	elif rule_num == 4:
		board = rule4(board)
	elif rule_num == 5:
		board = rule5(board)
	elif rule_num == 6:
		board = rule6(board)
	elif rule_num == 7:
		board = rule7(board)
	elif rule_num == 8:
		board = rule8(board)
	elif rule_num == 9:
		board = rule9(board)
	elif rule_num == 10:
		board = rule10(board)
	elif rule_num == 11:
		board = rule11(board)
	elif rule_num == 12:
		board = rule12(board)
	elif rule_num == 13:
		board = rule13(board)
	elif rule_num == 14:
		board = rule14(board)
	elif rule_num == 15:
		board = rule15(board)
	elif rule_num == 16:
		board = rule16(board)
	elif rule_num == 17:
		board = rule17(board)
	elif rule_num == 18:
		board = rule18(board)
	elif rule_num == 19:
		board = rule19(board)
	elif rule_num == 20:
		board = rule20(board)
	elif rule_num == 21:
		board = rule21(board)
	elif rule_num == 22:
		board = rule22(board)
	elif rule_num == 23:
		board = rule23(board)
	elif rule_num == 24:
		board = rule24(board)
	elif rule_num == 25:
		board = rule25(board)
	elif rule_num == 26:
		board = rule26(board)
	elif rule_num == 27:
		board = rule27(board)
	elif rule_num == 28:
		board = rule28(board)
	elif rule_num == 29:
		board = rule29(board)
	elif rule_num == 30:
		board = rule30(board)
	elif rule_num == 31:
		board = rule31(board)
	elif rule_num == 32:
		board = rule32(board)
	elif rule_num == 33:
		board = rule33(board)
	elif rule_num == 34:
		board = rule34(board)
	elif rule_num == 35:
		board = rule35(board)
	elif rule_num == 36:
		board = rule36(board)
	elif rule_num == 37:
		board = rule37(board)
	elif rule_num == 38:
		board = rule38(board)
	elif rule_num == 39:
		board = rule39(board)
	elif rule_num == 40:
		board = rule40(board)
	elif rule_num == 41:
		board = rule41(board)
	elif rule_num == 42:
		board = rule42(board)
	elif rule_num == 43:
		board = rule43(board)
	elif rule_num == 44:
		board = rule44(board)
	elif rule_num == 45:
		board = rule45(board)
	elif rule_num == 46:
		board = rule46(board)
	elif rule_num == 47:
		board = rule47(board)
	elif rule_num == 48:
		board = rule48(board)
	elif rule_num == 49:
		board = rule49(board)
	elif rule_num == 50:
		board = rule50(board)
	elif rule_num == 51:
		board = rule51(board)
	elif rule_num == 52:
		board = rule52(board)
	elif rule_num == 53:
		board = rule53(board)
	elif rule_num == 54:
		board = rule54(board)
	elif rule_num == 55:
		board = rule55(board)
	elif rule_num == 56:
		board = rule56(board)
	elif rule_num == 57:
		board = rule57(board)
	elif rule_num == 58:
		board = rule58(board)
	elif rule_num == 59:
		board = rule59(board)
	elif rule_num == 60:
		board = rule60(board)
	elif rule_num == 61:
		board = rule61(board)
	elif rule_num == 62:
		board = rule62(board)
	elif rule_num == 63:
		board = rule63(board)
	elif rule_num == 64:
		board = rule64(board)
	elif rule_num == 65:
		board = rule65(board)
	elif rule_num == 66:
		board = rule66(board)
	elif rule_num == 67:
		board = rule67(board)
	elif rule_num == 68:
		board = rule68(board)
	elif rule_num == 69:
		board = rule69(board)
	elif rule_num == 70:
		board = rule70(board)
	elif rule_num == 71:
		board = rule71(board)
	elif rule_num == 72:
		board = rule72(board)
	elif rule_num == 73:
		board = rule73(board)
	elif rule_num == 74:
		board = rule74(board)
	elif rule_num == 75:
		board = rule75(board)
	elif rule_num == 76:
		board = rule76(board)
	elif rule_num == 77:
		board = rule77(board)
	elif rule_num == 78:
		board = rule78(board)
	elif rule_num == 79:
		board = rule79(board)
	elif rule_num == 80:
		board = rule80(board)
	elif rule_num == 81:
		board = rule81(board)
	elif rule_num == 82:
		board = rule82(board)
	elif rule_num == 83:
		board = rule83(board)
	elif rule_num == 84:
		board = rule84(board)
	elif rule_num == 85:
		board = rule85(board)
	elif rule_num == 86:
		board = rule86(board)
	elif rule_num == 87:
		board = rule87(board)
	elif rule_num == 88:
		board = rule88(board)
	elif rule_num == 89:
		board = rule89(board)
	elif rule_num == 90:
		board = rule90(board)
	elif rule_num == 91:
		board = rule91(board)
	elif rule_num == 92:
		board = rule92(board)
	elif rule_num == 93:
		board = rule93(board)
	elif rule_num == 94:
		board = rule94(board)
	elif rule_num == 95:
		board = rule95(board)
	elif rule_num == 96:
		board = rule96(board)
	elif rule_num == 97:
		board = rule97(board)
	elif rule_num == 98:
		board = rule98(board)
	elif rule_num == 99:
		board = rule99(board)
	elif rule_num == 100:
		board = rule100(board)
	elif rule_num == 101:
		board = rule101(board)
	elif rule_num == 102:
		board = rule102(board)
	elif rule_num == 103:
		board = rule103(board)
	elif rule_num == 104:
		board = rule104(board)
	elif rule_num == 105:
		board = rule105(board)
	elif rule_num == 106:
		board = rule106(board)
	elif rule_num == 107:
		board = rule107(board)
	elif rule_num == 108:
		board = rule108(board)
	elif rule_num == 109:
		board = rule109(board)
	elif rule_num == 110:
		board = rule110(board)
	elif rule_num == 111:
		board = rule111(board)
	elif rule_num == 112:
		board = rule112(board)
	elif rule_num == 113:
		board = rule113(board)
	elif rule_num == 114:
		board = rule114(board)
	elif rule_num == 115:
		board = rule115(board)
	elif rule_num == 116:
		board = rule116(board)
	elif rule_num == 117:
		board = rule117(board)
	elif rule_num == 118:
		board = rule118(board)
	elif rule_num == 119:
		board = rule119(board)
	elif rule_num == 120:
		board = rule120(board)
	elif rule_num == 121:
		board = rule121(board)
	elif rule_num == 122:
		board = rule122(board)
	elif rule_num == 123:
		board = rule123(board)
	elif rule_num == 124:
		board = rule124(board)
	elif rule_num == 125:
		board = rule125(board)
	elif rule_num == 126:
		board = rule126(board)
	elif rule_num == 127:
		board = rule127(board)
	elif rule_num == 128:
		board = rule128(board)
	elif rule_num == 129:
		board = rule129(board)
	elif rule_num == 130:
		board = rule130(board)
	elif rule_num == 131:
		board = rule131(board)
	elif rule_num == 132:
		board = rule132(board)
	elif rule_num == 133:
		board = rule133(board)
	elif rule_num == 134:
		board = rule134(board)
	elif rule_num == 135:
		board = rule135(board)
	elif rule_num == 136:
		board = rule136(board)
	elif rule_num == 137:
		board = rule137(board)
	elif rule_num == 138:
		board = rule138(board)
	elif rule_num == 139:
		board = rule139(board)
	elif rule_num == 140:
		board = rule140(board)
	elif rule_num == 141:
		board = rule141(board)
	elif rule_num == 142:
		board = rule142(board)
	elif rule_num == 143:
		board = rule143(board)
	elif rule_num == 144:
		board = rule144(board)
	elif rule_num == 145:
		board = rule145(board)
	elif rule_num == 146:
		board = rule146(board)
	elif rule_num == 147:
		board = rule147(board)
	elif rule_num == 148:
		board = rule148(board)
	elif rule_num == 149:
		board = rule149(board)
	elif rule_num == 150:
		board = rule150(board)
	elif rule_num == 151:
		board = rule151(board)
	elif rule_num == 152:
		board = rule152(board)
	elif rule_num == 153:
		board = rule153(board)
	elif rule_num == 154:
		board = rule154(board)
	elif rule_num == 155:
		board = rule155(board)
	elif rule_num == 156:
		board = rule156(board)
	elif rule_num == 157:
		board = rule157(board)
	elif rule_num == 158:
		board = rule158(board)
	elif rule_num == 159:
		board = rule159(board)
	elif rule_num == 160:
		board = rule160(board)
	elif rule_num == 161:
		board = rule161(board)
	elif rule_num == 162:
		board = rule162(board)
	elif rule_num == 163:
		board = rule163(board)
	elif rule_num == 164:
		board = rule164(board)
	elif rule_num == 165:
		board = rule165(board)
	elif rule_num == 166:
		board = rule166(board)
	elif rule_num == 167:
		board = rule167(board)
	elif rule_num == 168:
		board = rule168(board)
	elif rule_num == 169:
		board = rule169(board)
	elif rule_num == 170:
		board = rule170(board)
	elif rule_num == 171:
		board = rule171(board)
	elif rule_num == 172:
		board = rule172(board)
	elif rule_num == 173:
		board = rule173(board)
	elif rule_num == 174:
		board = rule174(board)
	elif rule_num == 175:
		board = rule175(board)
	elif rule_num == 176:
		board = rule176(board)
	elif rule_num == 177:
		board = rule177(board)
	elif rule_num == 178:
		board = rule178(board)
	elif rule_num == 179:
		board = rule179(board)
	elif rule_num == 180:
		board = rule180(board)
	elif rule_num == 181:
		board = rule181(board)
	elif rule_num == 182:
		board = rule182(board)
	elif rule_num == 183:
		board = rule183(board)
	elif rule_num == 184:
		board = rule184(board)
	elif rule_num == 185:
		board = rule185(board)
	elif rule_num == 186:
		board = rule186(board)
	elif rule_num == 187:
		board = rule187(board)
	elif rule_num == 188:
		board = rule188(board)
	elif rule_num == 189:
		board = rule189(board)
	elif rule_num == 190:
		board = rule190(board)
	elif rule_num == 191:
		board = rule191(board)
	elif rule_num == 192:
		board = rule192(board)
	elif rule_num == 193:
		board = rule193(board)
	elif rule_num == 194:
		board = rule194(board)
	elif rule_num == 195:
		board = rule195(board)
	elif rule_num == 196:
		board = rule196(board)
	elif rule_num == 197:
		board = rule197(board)
	elif rule_num == 198:
		board = rule198(board)
	elif rule_num == 199:
		board = rule199(board)
	elif rule_num == 200:
		board = rule200(board)
	elif rule_num == 201:
		board = rule201(board)
	elif rule_num == 202:
		board = rule202(board)
	elif rule_num == 203:
		board = rule203(board)
	elif rule_num == 204:
		board = rule204(board)
	elif rule_num == 205:
		board = rule205(board)
	elif rule_num == 206:
		board = rule206(board)
	elif rule_num == 207:
		board = rule207(board)
	elif rule_num == 208:
		board = rule208(board)
	elif rule_num == 209:
		board = rule209(board)
	elif rule_num == 210:
		board = rule210(board)
	elif rule_num == 211:
		board = rule211(board)
	elif rule_num == 212:
		board = rule212(board)
	elif rule_num == 213:
		board = rule213(board)
	elif rule_num == 214:
		board = rule214(board)
	elif rule_num == 215:
		board = rule215(board)
	elif rule_num == 216:
		board = rule216(board)
	elif rule_num == 217:
		board = rule217(board)
	elif rule_num == 218:
		board = rule218(board)
	elif rule_num == 219:
		board = rule219(board)
	elif rule_num == 220:
		board = rule220(board)
	elif rule_num == 221:
		board = rule221(board)
	elif rule_num == 222:
		board = rule222(board)
	elif rule_num == 223:
		board = rule223(board)
	elif rule_num == 224:
		board = rule224(board)
	elif rule_num == 225:
		board = rule225(board)
	elif rule_num == 226:
		board = rule226(board)
	elif rule_num == 227:
		board = rule227(board)
	elif rule_num == 228:
		board = rule228(board)
	elif rule_num == 229:
		board = rule229(board)
	elif rule_num == 230:
		board = rule230(board)
	elif rule_num == 231:
		board = rule231(board)
	elif rule_num == 232:
		board = rule232(board)
	elif rule_num == 233:
		board = rule233(board)
	elif rule_num == 234:
		board = rule234(board)
	elif rule_num == 235:
		board = rule235(board)
	elif rule_num == 236:
		board = rule236(board)
	elif rule_num == 237:
		board = rule237(board)
	elif rule_num == 238:
		board = rule238(board)
	elif rule_num == 239:
		board = rule239(board)
	elif rule_num == 240:
		board = rule240(board)
	elif rule_num == 241:
		board = rule241(board)
	elif rule_num == 242:
		board = rule242(board)
	elif rule_num == 243:
		board = rule243(board)
	elif rule_num == 244:
		board = rule244(board)
	elif rule_num == 245:
		board = rule245(board)
	elif rule_num == 246:
		board = rule246(board)
	elif rule_num == 247:
		board = rule247(board)
	elif rule_num == 248:
		board = rule248(board)
	elif rule_num == 249:
		board = rule249(board)
	elif rule_num == 250:
		board = rule250(board)
	elif rule_num == 251:
		board = rule251(board)
	elif rule_num == 252:
		board = rule252(board)
	elif rule_num == 253:
		board = rule253(board)
	elif rule_num == 254:
		board = rule254(board)
	elif rule_num == 255:
		board = rule255(board)

	board = board.astype(int)
	sum_vert = np.sum(board, axis=0)# width
	sum_hori = np.sum(board, axis=1)# height ~time

	if plot == True:
		plt.figure(figsize=fig_sz1)
		plt.title('Rule '+str(rule_num))
		plt.imshow(board, cmap = 'binary')
		if type(save_path) != type(None):
			name = save_path+'rule'+str(rule_num)+'_sz'+str(shape[0])
			name = name+'-'+str(shape[0])
			if (start == 'center')|(type(seed) == None):
				name = name+'_simpl'
			elif (start == 'bin_w')|(type(seed) == None):
				name = name+'_bin_w'
			elif (start == 'bin_k')|(type(seed) == None):
				name = name+'_bin_k'
			elif (start == 'rand')|(type(seed) == None):
				name = name+'_rand'
			elif type(seed) != None:
				name = name+'_self'

			plt.savefig(name, bbox_inches='tight')
		plt.show()

		plt.figure(figsize=fig_sz2)
		plt.subplot(2, 1, 1)
		plt.title('Rule '+str(rule_num))
		plt.grid(True)
		plt.plot(sum_vert)
		plt.xlabel('Lines')
		plt.ylabel("Sum over the columns")
		plt.subplot(2, 1, 2)
		plt.title('Rule '+str(rule_num))
		plt.grid(True)
		plt.plot(sum_hori)
		plt.xlabel('Columns')
		plt.ylabel('Sum over the lines')
		if type(save_path) != type(None):
			name = save_path+'rule'+str(rule_num)+'_sz'+str(shape[0])
			name = name+'-'+str(shape[0])
			if (start == 'center')|(type(seed) == None):
				name = name+'_simpl.png'
			elif (start == 'bin_w')|(type(seed) == None):
				name = name+'_bin_w.png'
			elif (start == 'bin_k')|(type(seed) == None):
				name = name+'_bin_k.png'
			elif (start == 'rand')|(type(seed) == None):
				name = name+'_rand.png'
			elif type(seed) != None:
				name = name+'_self.png'
				
			plt.savefig(name, bbox_inches='tight')

		plt.show()

	return board, sum_vert, sum_hori
