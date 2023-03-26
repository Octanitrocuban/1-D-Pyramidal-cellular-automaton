# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 13:12:40 2023

@author: Matthieu Nougaret
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


def ApplicationRules(Shape, RuleN, Start='center', Seed=None, Plot=True,
					 SavePath=None, FgSz1=(10, 10), FgSz2=(15, 10)):
	"""
	Function to applied one of the 256 possible rules.

	Parameters
	----------
	Shape : tuple or int
		Shape/size of the array that will be used for the cellular automata.
	RuleN : int
		The number of the rule that will be used.
	Start : str, optional
		Indicate how the the cellular automata will be initialized. The
		default is 'center'.
	Seed : NoneType, optional
		Parameter to make its own initialisaion of the automata. It must have
		the same length as the width indicate for the 'Shape' parameter. It
		must be a list or numpy.ndarray type. If NoneType, the initialisation
		of the automata will follow the 'Start' parameter indication. The
		default is None.
	Plot : bool, optional
		Indicate if the result will be ploted. The default is True.
	SavePath : str, optional
		Path to where the plot will be saved. The default is None.
	FgSz1 : tuple, optional
		Size of the figure that will show the final state of the automata. The
		default is (10, 10).
	FgSz2 : tuple, optional
		Size of the figure that will show the evolution of the automata state.
		The default is (15, 10).

	Raises
	------
	ValueError
		'Start' parameter is not from the possible list.
	ValueError
		The asked rule doesn't exist.

	Returns
	-------
	Table : numpy.ndarray
		Final state of the automata.
	SumVert : numpy.ndarray
		Sum of the final state over the axis 0.
	SumHori : numpy.ndarray
		Sum of the final state over the axis 1.

	"""
	if (type(Shape) == int)|(type(Shape) == float):
		Shape = (int(Shape), int(Shape))

	Table = np.zeros(Shape)

	if Start == 'center':
		Table[0, Shape[1]//2] = 1
	elif Start == 'bin_w':
		Table[0, (np.arange(Shape[1])%2) == 0] = 1
	elif Start == 'bin_k':
		Table[0, (np.arange(Shape[1])%2) == 1] = 1
	elif Start == 'rand':
		Table[0] = np.random.randint(0, 2, Shape[1])
	else:
		raise ValueError('Start parameter must be from the list: ["center",'+
						 ' "bin_w", "bin_k", "rand"]. Get '+str(Start))

	if type(Seed) == np.ndarray:
		Table[0] = Seed

	List = list(range(256))
	if RuleN not in List:
		raise ValueError("There are 256 rules, from 0 to 255. Get "+str(RuleN))

	if RuleN == 0:
		Table = Rule0(Table)
	elif RuleN == 1:
		Table = Rule1(Table)
	elif RuleN == 2:
		Table = Rule2(Table)
	elif RuleN == 3:
		Table = Rule3(Table)
	elif RuleN == 4:
		Table = Rule4(Table)
	elif RuleN == 5:
		Table = Rule5(Table)
	elif RuleN == 6:
		Table = Rule6(Table)
	elif RuleN == 7:
		Table = Rule7(Table)
	elif RuleN == 8:
		Table = Rule8(Table)
	elif RuleN == 9:
		Table = Rule9(Table)
	elif RuleN == 10:
		Table = Rule10(Table)
	elif RuleN == 11:
		Table = Rule11(Table)
	elif RuleN == 12:
		Table = Rule12(Table)
	elif RuleN == 13:
		Table = Rule13(Table)
	elif RuleN == 14:
		Table = Rule14(Table)
	elif RuleN == 15:
		Table = Rule15(Table)
	elif RuleN == 16:
		Table = Rule16(Table)
	elif RuleN == 17:
		Table = Rule17(Table)
	elif RuleN == 18:
		Table = Rule18(Table)
	elif RuleN == 19:
		Table = Rule19(Table)
	elif RuleN == 20:
		Table = Rule20(Table)
	elif RuleN == 21:
		Table = Rule21(Table)
	elif RuleN == 22:
		Table = Rule22(Table)
	elif RuleN == 23:
		Table = Rule23(Table)
	elif RuleN == 24:
		Table = Rule24(Table)
	elif RuleN == 25:
		Table = Rule25(Table)
	elif RuleN == 26:
		Table = Rule26(Table)
	elif RuleN == 27:
		Table = Rule27(Table)
	elif RuleN == 28:
		Table = Rule28(Table)
	elif RuleN == 29:
		Table = Rule29(Table)
	elif RuleN == 30:
		Table = Rule30(Table)
	elif RuleN == 31:
		Table = Rule31(Table)
	elif RuleN == 32:
		Table = Rule32(Table)
	elif RuleN == 33:
		Table = Rule33(Table)
	elif RuleN == 34:
		Table = Rule34(Table)
	elif RuleN == 35:
		Table = Rule35(Table)
	elif RuleN == 36:
		Table = Rule36(Table)
	elif RuleN == 37:
		Table = Rule37(Table)
	elif RuleN == 38:
		Table = Rule38(Table)
	elif RuleN == 39:
		Table = Rule39(Table)
	elif RuleN == 40:
		Table = Rule40(Table)
	elif RuleN == 41:
		Table = Rule41(Table)
	elif RuleN == 42:
		Table = Rule42(Table)
	elif RuleN == 43:
		Table = Rule43(Table)
	elif RuleN == 44:
		Table = Rule44(Table)
	elif RuleN == 45:
		Table = Rule45(Table)
	elif RuleN == 46:
		Table = Rule46(Table)
	elif RuleN == 47:
		Table = Rule47(Table)
	elif RuleN == 48:
		Table = Rule48(Table)
	elif RuleN == 49:
		Table = Rule49(Table)
	elif RuleN == 50:
		Table = Rule50(Table)
	elif RuleN == 51:
		Table = Rule51(Table)
	elif RuleN == 52:
		Table = Rule52(Table)
	elif RuleN == 53:
		Table = Rule53(Table)
	elif RuleN == 54:
		Table = Rule54(Table)
	elif RuleN == 55:
		Table = Rule55(Table)
	elif RuleN == 56:
		Table = Rule56(Table)
	elif RuleN == 57:
		Table = Rule57(Table)
	elif RuleN == 58:
		Table = Rule58(Table)
	elif RuleN == 59:
		Table = Rule59(Table)
	elif RuleN == 60:
		Table = Rule60(Table)
	elif RuleN == 61:
		Table = Rule61(Table)
	elif RuleN == 62:
		Table = Rule62(Table)
	elif RuleN == 63:
		Table = Rule63(Table)
	elif RuleN == 64:
		Table = Rule64(Table)
	elif RuleN == 65:
		Table = Rule65(Table)
	elif RuleN == 66:
		Table = Rule66(Table)
	elif RuleN == 67:
		Table = Rule67(Table)
	elif RuleN == 68:
		Table = Rule68(Table)
	elif RuleN == 69:
		Table = Rule69(Table)
	elif RuleN == 70:
		Table = Rule70(Table)
	elif RuleN == 71:
		Table = Rule71(Table)
	elif RuleN == 72:
		Table = Rule72(Table)
	elif RuleN == 73:
		Table = Rule73(Table)
	elif RuleN == 74:
		Table = Rule74(Table)
	elif RuleN == 75:
		Table = Rule75(Table)
	elif RuleN == 76:
		Table = Rule76(Table)
	elif RuleN == 77:
		Table = Rule77(Table)
	elif RuleN == 78:
		Table = Rule78(Table)
	elif RuleN == 79:
		Table = Rule79(Table)
	elif RuleN == 80:
		Table = Rule80(Table)
	elif RuleN == 81:
		Table = Rule81(Table)
	elif RuleN == 82:
		Table = Rule82(Table)
	elif RuleN == 83:
		Table = Rule83(Table)
	elif RuleN == 84:
		Table = Rule84(Table)
	elif RuleN == 85:
		Table = Rule85(Table)
	elif RuleN == 86:
		Table = Rule86(Table)
	elif RuleN == 87:
		Table = Rule87(Table)
	elif RuleN == 88:
		Table = Rule88(Table)
	elif RuleN == 89:
		Table = Rule89(Table)
	elif RuleN == 90:
		Table = Rule90(Table)
	elif RuleN == 91:
		Table = Rule91(Table)
	elif RuleN == 92:
		Table = Rule92(Table)
	elif RuleN == 93:
		Table = Rule93(Table)
	elif RuleN == 94:
		Table = Rule94(Table)
	elif RuleN == 95:
		Table = Rule95(Table)
	elif RuleN == 96:
		Table = Rule96(Table)
	elif RuleN == 97:
		Table = Rule97(Table)
	elif RuleN == 98:
		Table = Rule98(Table)
	elif RuleN == 99:
		Table = Rule99(Table)
	elif RuleN == 100:
		Table = Rule100(Table)
	elif RuleN == 101:
		Table = Rule101(Table)
	elif RuleN == 102:
		Table = Rule102(Table)
	elif RuleN == 103:
		Table = Rule103(Table)
	elif RuleN == 104:
		Table = Rule104(Table)
	elif RuleN == 105:
		Table = Rule105(Table)
	elif RuleN == 106:
		Table = Rule106(Table)
	elif RuleN == 107:
		Table = Rule107(Table)
	elif RuleN == 108:
		Table = Rule108(Table)
	elif RuleN == 109:
		Table = Rule109(Table)
	elif RuleN == 110:
		Table = Rule110(Table)
	elif RuleN == 111:
		Table = Rule111(Table)
	elif RuleN == 112:
		Table = Rule112(Table)
	elif RuleN == 113:
		Table = Rule113(Table)
	elif RuleN == 114:
		Table = Rule114(Table)
	elif RuleN == 115:
		Table = Rule115(Table)
	elif RuleN == 116:
		Table = Rule116(Table)
	elif RuleN == 117:
		Table = Rule117(Table)
	elif RuleN == 118:
		Table = Rule118(Table)
	elif RuleN == 119:
		Table = Rule119(Table)
	elif RuleN == 120:
		Table = Rule120(Table)
	elif RuleN == 121:
		Table = Rule121(Table)
	elif RuleN == 122:
		Table = Rule122(Table)
	elif RuleN == 123:
		Table = Rule123(Table)
	elif RuleN == 124:
		Table = Rule124(Table)
	elif RuleN == 125:
		Table = Rule125(Table)
	elif RuleN == 126:
		Table = Rule126(Table)
	elif RuleN == 127:
		Table = Rule127(Table)
	elif RuleN == 128:
		Table = Rule128(Table)
	elif RuleN == 129:
		Table = Rule129(Table)
	elif RuleN == 130:
		Table = Rule130(Table)
	elif RuleN == 131:
		Table = Rule131(Table)
	elif RuleN == 132:
		Table = Rule132(Table)
	elif RuleN == 133:
		Table = Rule133(Table)
	elif RuleN == 134:
		Table = Rule134(Table)
	elif RuleN == 135:
		Table = Rule135(Table)
	elif RuleN == 136:
		Table = Rule136(Table)
	elif RuleN == 137:
		Table = Rule137(Table)
	elif RuleN == 138:
		Table = Rule138(Table)
	elif RuleN == 139:
		Table = Rule139(Table)
	elif RuleN == 140:
		Table = Rule140(Table)
	elif RuleN == 141:
		Table = Rule141(Table)
	elif RuleN == 142:
		Table = Rule142(Table)
	elif RuleN == 143:
		Table = Rule143(Table)
	elif RuleN == 144:
		Table = Rule144(Table)
	elif RuleN == 145:
		Table = Rule145(Table)
	elif RuleN == 146:
		Table = Rule146(Table)
	elif RuleN == 147:
		Table = Rule147(Table)
	elif RuleN == 148:
		Table = Rule148(Table)
	elif RuleN == 149:
		Table = Rule149(Table)
	elif RuleN == 150:
		Table = Rule150(Table)
	elif RuleN == 151:
		Table = Rule151(Table)
	elif RuleN == 152:
		Table = Rule152(Table)
	elif RuleN == 153:
		Table = Rule153(Table)
	elif RuleN == 154:
		Table = Rule154(Table)
	elif RuleN == 155:
		Table = Rule155(Table)
	elif RuleN == 156:
		Table = Rule156(Table)
	elif RuleN == 157:
		Table = Rule157(Table)
	elif RuleN == 158:
		Table = Rule158(Table)
	elif RuleN == 159:
		Table = Rule159(Table)
	elif RuleN == 160:
		Table = Rule160(Table)
	elif RuleN == 161:
		Table = Rule161(Table)
	elif RuleN == 162:
		Table = Rule162(Table)
	elif RuleN == 163:
		Table = Rule163(Table)
	elif RuleN == 164:
		Table = Rule164(Table)
	elif RuleN == 165:
		Table = Rule165(Table)
	elif RuleN == 166:
		Table = Rule166(Table)
	elif RuleN == 167:
		Table = Rule167(Table)
	elif RuleN == 168:
		Table = Rule168(Table)
	elif RuleN == 169:
		Table = Rule169(Table)
	elif RuleN == 170:
		Table = Rule170(Table)
	elif RuleN == 171:
		Table = Rule171(Table)
	elif RuleN == 172:
		Table = Rule172(Table)
	elif RuleN == 173:
		Table = Rule173(Table)
	elif RuleN == 174:
		Table = Rule174(Table)
	elif RuleN == 175:
		Table = Rule175(Table)
	elif RuleN == 176:
		Table = Rule176(Table)
	elif RuleN == 177:
		Table = Rule177(Table)
	elif RuleN == 178:
		Table = Rule178(Table)
	elif RuleN == 179:
		Table = Rule179(Table)
	elif RuleN == 180:
		Table = Rule180(Table)
	elif RuleN == 181:
		Table = Rule181(Table)
	elif RuleN == 182:
		Table = Rule182(Table)
	elif RuleN == 183:
		Table = Rule183(Table)
	elif RuleN == 184:
		Table = Rule184(Table)
	elif RuleN == 185:
		Table = Rule185(Table)
	elif RuleN == 186:
		Table = Rule186(Table)
	elif RuleN == 187:
		Table = Rule187(Table)
	elif RuleN == 188:
		Table = Rule188(Table)
	elif RuleN == 189:
		Table = Rule189(Table)
	elif RuleN == 190:
		Table = Rule190(Table)
	elif RuleN == 191:
		Table = Rule191(Table)
	elif RuleN == 192:
		Table = Rule192(Table)
	elif RuleN == 193:
		Table = Rule193(Table)
	elif RuleN == 194:
		Table = Rule194(Table)
	elif RuleN == 195:
		Table = Rule195(Table)
	elif RuleN == 196:
		Table = Rule196(Table)
	elif RuleN == 197:
		Table = Rule197(Table)
	elif RuleN == 198:
		Table = Rule198(Table)
	elif RuleN == 199:
		Table = Rule199(Table)
	elif RuleN == 200:
		Table = Rule200(Table)
	elif RuleN == 201:
		Table = Rule201(Table)
	elif RuleN == 202:
		Table = Rule202(Table)
	elif RuleN == 203:
		Table = Rule203(Table)
	elif RuleN == 204:
		Table = Rule204(Table)
	elif RuleN == 205:
		Table = Rule205(Table)
	elif RuleN == 206:
		Table = Rule206(Table)
	elif RuleN == 207:
		Table = Rule207(Table)
	elif RuleN == 208:
		Table = Rule208(Table)
	elif RuleN == 209:
		Table = Rule209(Table)
	elif RuleN == 210:
		Table = Rule210(Table)
	elif RuleN == 211:
		Table = Rule211(Table)
	elif RuleN == 212:
		Table = Rule212(Table)
	elif RuleN == 213:
		Table = Rule213(Table)
	elif RuleN == 214:
		Table = Rule214(Table)
	elif RuleN == 215:
		Table = Rule215(Table)
	elif RuleN == 216:
		Table = Rule216(Table)
	elif RuleN == 217:
		Table = Rule217(Table)
	elif RuleN == 218:
		Table = Rule218(Table)
	elif RuleN == 219:
		Table = Rule219(Table)
	elif RuleN == 220:
		Table = Rule220(Table)
	elif RuleN == 221:
		Table = Rule221(Table)
	elif RuleN == 222:
		Table = Rule222(Table)
	elif RuleN == 223:
		Table = Rule223(Table)
	elif RuleN == 224:
		Table = Rule224(Table)
	elif RuleN == 225:
		Table = Rule225(Table)
	elif RuleN == 226:
		Table = Rule226(Table)
	elif RuleN == 227:
		Table = Rule227(Table)
	elif RuleN == 228:
		Table = Rule228(Table)
	elif RuleN == 229:
		Table = Rule229(Table)
	elif RuleN == 230:
		Table = Rule230(Table)
	elif RuleN == 231:
		Table = Rule231(Table)
	elif RuleN == 232:
		Table = Rule232(Table)
	elif RuleN == 233:
		Table = Rule233(Table)
	elif RuleN == 234:
		Table = Rule234(Table)
	elif RuleN == 235:
		Table = Rule235(Table)
	elif RuleN == 236:
		Table = Rule236(Table)
	elif RuleN == 237:
		Table = Rule237(Table)
	elif RuleN == 238:
		Table = Rule238(Table)
	elif RuleN == 239:
		Table = Rule239(Table)
	elif RuleN == 240:
		Table = Rule240(Table)
	elif RuleN == 241:
		Table = Rule241(Table)
	elif RuleN == 242:
		Table = Rule242(Table)
	elif RuleN == 243:
		Table = Rule243(Table)
	elif RuleN == 244:
		Table = Rule244(Table)
	elif RuleN == 245:
		Table = Rule245(Table)
	elif RuleN == 246:
		Table = Rule246(Table)
	elif RuleN == 247:
		Table = Rule247(Table)
	elif RuleN == 248:
		Table = Rule248(Table)
	elif RuleN == 249:
		Table = Rule249(Table)
	elif RuleN == 250:
		Table = Rule250(Table)
	elif RuleN == 251:
		Table = Rule251(Table)
	elif RuleN == 252:
		Table = Rule252(Table)
	elif RuleN == 253:
		Table = Rule253(Table)
	elif RuleN == 254:
		Table = Rule254(Table)
	elif RuleN == 255:
		Table = Rule255(Table)

	SumVert = np.sum(Table, axis=0)# Width
	SumHori = np.sum(Table, axis=1)# ~Time

	if Plot == True:
		plt.figure(figsize=FgSz1)
		plt.title('Rule '+str(RuleN))
		plt.imshow(Table, cmap = 'binary')
		if type(SavePath) != type(None):
			name = SavePath+'rule'+str(RuleN)+'_sz'+str(Shape[0])
			name = name+'-'+str(Shape[0])
			if (Start == 'center')|(type(Seed) == None):
				name = name+'_simpl'
			elif (Start == 'bin_w')|(type(Seed) == None):
				name = name+'_bin_w'
			elif (Start == 'bin_k')|(type(Seed) == None):
				name = name+'_bin_k'
			elif (Start == 'rand')|(type(Seed) == None):
				name = name+'_rand'
			elif type(Seed) != None:
				name = name+'_self'

			plt.savefig(name, bbox_inches='tight')
		plt.show()

		plt.figure(figsize=FgSz2)
		plt.subplot(2, 1, 1)
		plt.title('Rule '+str(RuleN))
		plt.grid(True)
		plt.plot(SumVert)
		plt.xlabel('Lines')
		plt.ylabel("Sum over the columns")
		plt.subplot(2, 1, 2)
		plt.title('Rule '+str(RuleN))
		plt.grid(True)
		plt.plot(SumHori)
		plt.xlabel('Columns')
		plt.ylabel('Sum over the lines')
		if type(SavePath) != type(None):
			name = SavePath+'rule'+str(RuleN)+'_sz'+str(Shape[0])
			name = name+'-'+str(Shape[0])
			if (Start == 'center')|(type(Seed) == None):
				name = name+'_simpl.png'
			elif (Start == 'bin_w')|(type(Seed) == None):
				name = name+'_bin_w.png'
			elif (Start == 'bin_k')|(type(Seed) == None):
				name = name+'_bin_k.png'
			elif (Start == 'rand')|(type(Seed) == None):
				name = name+'_rand.png'
			elif type(Seed) != None:
				name = name+'_self.png'
				
			plt.savefig(name, bbox_inches='tight')
		plt.show()

	return Table, SumVert, SumHori
