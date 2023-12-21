# -*- coding: utf-8 -*-
"""
This module contain the function to use from 100 to 199-th rule.
"""
import numpy as np
from embedding_func import recadre_pyr
#=============================================================================
def rule100(board, padd=0):
	"""
	Function to applie the 100-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 100-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 100-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule100(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule101(board, padd=0):
	"""
	Function to applie the 101-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 101-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 101-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule101(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
					[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
					[1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
					[0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule102(board, padd=0):
	"""
	Function to applie the 102-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 102-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 102-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule102(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
					[0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
					[1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
					[0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
					[1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
					[0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule103(board, padd=0):
	"""
	Function to applie the 103-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 103-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 103-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule103(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule104(board, padd=0):
	"""
	Function to applie the 104-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 104-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 104-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule104(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
					[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule105(board, padd=0):
	"""
	Function to applie the 105-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 105-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 105-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule105(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule106(board, padd=0):
	"""
	Function to applie the 106-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 106-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 106-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule106(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
					[1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
					[0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
					[1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
					[1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
					[0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule107(board, padd=0):
	"""
	Function to applie the 107-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 107-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 107-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule107(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule108(board, padd=0):
	"""
	Function to applie the 108-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 108-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 108-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule108(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0],
					[1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0],
					[1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1],
					[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule109(board, padd=0):
	"""
	Function to applie the 109-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 109-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 109-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule109(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
					[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule110(board, padd=0):
	"""
	Function to applie the 110-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 110-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 110-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule110(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
					[1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
					[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
					[1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1],
					[1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule111(board, padd=0):
	"""
	Function to applie the 111-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 111-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 111-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule111(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule112(board, padd=0):
	"""
	Function to applie the 112-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 112-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 112-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule112(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule113(board, padd=0):
	"""
	Function to applie the 113-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 113-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 113-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule113(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule114(board, padd=0):
	"""
	Function to applie the 114-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 114-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 114-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule114(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
					[1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
					[0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
					[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
					[0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
					[1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule115(board, padd=0):
	"""
	Function to applie the 115-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 115-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 115-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule115(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
					[1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule116(board, padd=0):
	"""
	Function to applie the 116-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 116-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 116-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule116(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule117(board, padd=0):
	"""
	Function to applie the 117-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 117-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 117-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule117(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
					[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
					[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule118(board, padd=0):
	"""
	Function to applie the 118-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 118-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 118-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule118(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1],
					[0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0],
					[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
					[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
					[0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule119(board, padd=0):
	"""
	Function to applie the 119-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 119-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 119-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule119(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
					[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule120(board, padd=0):
	"""
	Function to applie the 120-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 120-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 120-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule120(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
					[1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
					[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
					[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
					[0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
					[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule121(board, padd=0):
	"""
	Function to applie the 121-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 121-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 121-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule121(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
					[0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
					[1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
					[0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule122(board, padd=0):
	"""
	Function to applie the 122-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 122-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 122-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule122(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
					[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0],
					[0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
					[0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule123(board, padd=0):
	"""
	Function to applie the 123-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 123-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 123-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule123(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0],
					[1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
					[0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0],
					[0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],
					[0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule124(board, padd=0):
	"""
	Function to applie the 124-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 124-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 124-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule124(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
					[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule125(board, padd=0):
	"""
	Function to applie the 125-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 125-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 125-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule125(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
					[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule126(board, padd=0):
	"""
	Function to applie the 126-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 126-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 126-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule126(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule127(board, padd=0):
	"""
	Function to applie the 127-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 127-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 127-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule127(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
					[0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
					[0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
					[1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1],
					[0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule128(board, padd=0):
	"""
	Function to applie the 128-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 128-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 128-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule128(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
					[1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1],
					[0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
					[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
					[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
					[1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1],
					[0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule129(board, padd=0):
	"""
	Function to applie the 129-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 129-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 129-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule129(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule130(board, padd=0):
	"""
	Function to applie the 130-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 130-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 130-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule130(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule131(board, padd=0):
	"""
	Function to applie the 131-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 131-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 131-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule131(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule132(board, padd=0):
	"""
	Function to applie the 132-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 132-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 132-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule132(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
					[0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
					[1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule133(board, padd=0):
	"""
	Function to applie the 133-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 133-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 133-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule133(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule134(board, padd=0):
	"""
	Function to applie the 134-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 134-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 134-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule134(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule135(board, padd=0):
	"""
	Function to applie the 135-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 135-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 135-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule135(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule136(board, padd=0):
	"""
	Function to applie the 136-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 136-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 136-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule136(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule137(board, padd=0):
	"""
	Function to applie the 137-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 137-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 137-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule137(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule138(board, padd=0):
	"""
	Function to applie the 138-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 138-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 138-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule138(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
					[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
					[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule139(board, padd=0):
	"""
	Function to applie the 139-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 139-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 139-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule139(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule140(board, padd=0):
	"""
	Function to applie the 140-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 140-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 140-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule140(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule141(board, padd=0):
	"""
	Function to applie the 141-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 141-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 141-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule141(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule142(board, padd=0):
	"""
	Function to applie the 142-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 142-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 142-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule142(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule143(board, padd=0):
	"""
	Function to applie the 143-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 143-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 143-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule143(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule144(board, padd=0):
	"""
	Function to applie the 144-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 144-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 144-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule144(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule145(board, padd=0):
	"""
	Function to applie the 145-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 145-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 145-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule145(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule146(board, padd=0):
	"""
	Function to applie the 146-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 146-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 146-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule146(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule147(board, padd=0):
	"""
	Function to applie the 147-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 147-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 147-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule147(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule148(board, padd=0):
	"""
	Function to applie the 148-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 148-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 148-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule148(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule149(board, padd=0):
	"""
	Function to applie the 149-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 149-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 149-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule149(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule150(board, padd=0):
	"""
	Function to applie the 150-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 150-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 150-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule150(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule151(board, padd=0):
	"""
	Function to applie the 151-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 151-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 151-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule151(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule152(board, padd=0):
	"""
	Function to applie the 152-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 152-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 152-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule152(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule153(board, padd=0):
	"""
	Function to applie the 1513-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 153-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 153-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule153(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule154(board, padd=0):
	"""
	Function to applie the 154-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 154-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 154-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule154(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule155(board, padd=0):
	"""
	Function to applie the 155-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 155-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 155-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule155(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule156(board, padd=0):
	"""
	Function to applie the 156-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 156-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 156-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule156(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule157(board, padd=0):
	"""
	Function to applie the 157-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 157-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 157-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule157(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule158(board, padd=0):
	"""
	Function to applie the 158-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 158-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 158-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule158(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule159(board, padd=0):
	"""
	Function to applie the 159-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 159-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 159-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule159(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule160(board, padd=0):
	"""
	Function to applie the 160-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 160-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 160-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule160(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule161(board, padd=0):
	"""
	Function to applie the 161-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 161-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 161-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule161(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule162(board, padd=0):
	"""
	Function to applie the 162-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 162-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 162-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule162(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule163(board, padd=0):
	"""
	Function to applie the 163-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 163-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 163-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule163(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule164(board, padd=0):
	"""
	Function to applie the 164-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 164-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 164-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule164(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule165(board, padd=0):
	"""
	Function to applie the 165-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 165-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 165-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule165(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule166(board, padd=0):
	"""
	Function to applie the 166-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 166-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 166-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule166(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
					[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
					[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
					[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule167(board, padd=0):
	"""
	Function to applie the 167-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 167-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 167-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule167(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule168(board, padd=0):
	"""
	Function to applie the 168-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 168-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 168-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule168(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule169(board, padd=0):
	"""
	Function to applie the 169-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 169-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 169-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule169(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule170(board, padd=0):
	"""
	Function to applie the 170-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 170-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 170-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule170(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule171(board, padd=0):
	"""
	Function to applie the 171-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 171-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 171-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule171(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule172(board, padd=0):
	"""
	Function to applie the 172-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 172-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 172-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule172(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule173(board, padd=0):
	"""
	Function to applie the 173-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 173-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 173-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule173(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
					[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
					[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule174(board, padd=0):
	"""
	Function to applie the 174-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 174-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 174-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule174(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule175(board, padd=0):
	"""
	Function to applie the 175-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 175-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 175-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule175(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0],
					[0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
					[1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
					[0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
					[1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
					[0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule176(board, padd=0):
	"""
	Function to applie the 176-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 176-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 176-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule176(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule177(board, padd=0):
	"""
	Function to applie the 177-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 177-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 177-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule177(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
					[1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0],
					[0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0],
					[0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0],
					[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
					[1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
					[0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule178(board, padd=0):
	"""
	Function to applie the 178-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 178-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 178-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule178(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
					[1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule179(board, padd=0):
	"""
	Function to applie the 179-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 179-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 179-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule179(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
					[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1],
					[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
					[0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule180(board, padd=0):
	"""
	Function to applie the 180-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 180-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 180-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule180(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
					[0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule181(board, padd=0):
	"""
	Function to applie the 181-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 181-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 181-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule181(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule182(board, padd=0):
	"""
	Function to applie the 182-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 182-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 182-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule182(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0],
					[0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
					[0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1],
					[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					[1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
					[0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0],
					[0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule183(board, padd=0):
	"""
	Function to applie the 183-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 183-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 183-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule183(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule184(board, padd=0):
	"""
	Function to applie the 184-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 184-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 184-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule184(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
					[1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1],
					[0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0],
					[0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
					[1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
					[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],
					[0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
					[0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1],
					[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule185(board, padd=0):
	"""
	Function to applie the 185-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 185-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 185-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule185(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule186(board, padd=0):
	"""
	Function to applie the 186-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 186-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 186-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule186(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule187(board, padd=0):
	"""
	Function to applie the 187-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 187-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 187-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule187(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1],
					[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
					[0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
					[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule188(board, padd=0):
	"""
	Function to applie the 188-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 188-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 188-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule188(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule189(board, padd=0):
	"""
	Function to applie the 189-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 189-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 189-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule189(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule190(board, padd=0):
	"""
	Function to applie the 190-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 190-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 190-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule190(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
					[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
					[1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
					[1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
					[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
					[1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule191(board, padd=0):
	"""
	Function to applie the 191-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 191-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 191-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule191(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule192(board, padd=0):
	"""
	Function to applie the 192-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 192-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 192-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule192(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule193(board, padd=0):
	"""
	Function to applie the 193-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 193-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 193-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule193(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule194(board, padd=0):
	"""
	Function to applie the 194-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 194-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 194-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule194(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0],
					[1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
					[1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
					[0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule195(board, padd=0):
	"""
	Function to applie the 195-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 195-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 195-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule195(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule196(board, padd=0):
	"""
	Function to applie the 196-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 196-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 196-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule196(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],
					[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule197(board, padd=0):
	"""
	Function to applie the 197-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 197-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 197-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule197(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule198(board, padd=0):
	"""
	Function to applie the 198-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 198-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 198-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule198(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0],
					[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
					[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
					[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
					[1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule199(board, padd=0):
	"""
	Function to applie the 199-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 199-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 199-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule199(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
					[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]])

	"""
	copy_board = recadre_pyr(board, padd)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board
