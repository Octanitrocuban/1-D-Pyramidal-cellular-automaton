# -*- coding: utf-8 -*-
"""
This module contain the function to use from 240 to 255-th rule.
"""
import numpy as np
#=============================================================================
def recadre_pyr(board):
	"""
	Function to create a copy of the input array and adding a 0-padding to
	each side (columns) of it.

	Parameters
	----------
	board : numpy.ndarray
		The 2-dimensionals numpy.ndarry to be copyed and padded

	Returns
	-------
	copy : numpy.ndarray
		The 0-padded copy of the input array.

	Exemple
	-------
	In [0] : _ = np.ones((5, 5))
	In [1] : recadre_pyr(_)
	Out [2]: array([[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0],
					[0, 1, 1, 1, 1, 1, 0]])

	"""
	copy = np.zeros((board.shape[0], board.shape[1]+2))
	copy[:, 1:-1] = np.copy(board)
	return copy

def rule240(board):
	"""
	Function to applie the 240-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 240-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 240-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule240(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
					[1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule241(board):
	"""
	Function to applie the 241-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 241-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 241-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule241(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
					[0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule242(board):
	"""
	Function to applie the 242-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 242-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 242-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule242(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule243(board):
	"""
	Function to applie the 243-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 243-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 243-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule243(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
					[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
					[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule244(board):
	"""
	Function to applie the 244-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 244-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 244-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule244(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule245(board):
	"""
	Function to applie the 245-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 245-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 245-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule245(_)
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
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule246(board):
	"""
	Function to applie the 246-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 246-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 246-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule246(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule247(board):
	"""
	Function to applie the 247-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 247-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 247-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule247(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule248(board):
	"""
	Function to applie the 248-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 248-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 248-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule248(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],
					[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule249(board):
	"""
	Function to applie the 249-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 249-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 249-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule249(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule250(board):
	"""
	Function to applie the 250-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 250-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 250-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule250(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
					[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
					[1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule251(board):
	"""
	Function to applie the 251-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 251-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 251-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule251(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule252(board):
	"""
	Function to applie the 252-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 252-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 252-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule252(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule253(board):
	"""
	Function to applie the 253-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 253-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 253-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule253(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
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
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule254(board):
	"""
	Function to applie the 254-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 254-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 2542-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule254(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule255(board):
	"""
	Function to applie the 255-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 255-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 255-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule255(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

	"""
	copy_board = recadre_pyr(board)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, copy_board.shape[1]-1)+kernel
	for n in range(0, copy_board.shape[0]-1):
		condition = copy_board[n, ranje].T
		board[n+1, np.sum(condition == [0, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board
