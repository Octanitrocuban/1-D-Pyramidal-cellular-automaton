# -*- coding: utf-8 -*-
"""
module with matrix embending function.
"""
import numpy as np

def recadre_pyr(board, padd=0):
	"""
	Function to create a copy of the input array and adding a 0-padding to
	each side (columns) of it.

	Parameters
	----------
	board : numpy.ndarray
		The 2-dimensionals numpy.ndarry to be copyed and padded
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	copy : numpy.ndarray
		The 0-padded copy of the input array.

	Exemple
	-------
	In [0] : _ = np.ones((5, 5))
	In [1] : recadre_pyr(_, padd=0)
	Out [2]: array([[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0]])

	"""
	if type(padd) == int:
		copy = np.zeros((board.shape[0], board.shape[1]+2)) + padd
		copy[:, 1:-1] = np.copy(board)

	elif type(padd) == float:
		copy = np.ones((board.shape[0], board.shape[1]+2)) + padd
		copy[:, 1:-1] = np.copy(board)

	elif padd == 'cyclic':
		copy = np.zeros((board.shape[0], board.shape[1]+2))
		copy[:, 1:-1] = np.copy(board)
		copy[:,  0] = copy[:, -2]
		copy[:, -1] = copy[:,  1]

	elif padd == 'same':
		copy = np.zeros((board.shape[0], board.shape[1]+2))
		copy[:, 1:-1] = np.copy(board)
		copy[:,  0] = copy[:,  1]
		copy[:, -1] = copy[:, -2]

	return copy
