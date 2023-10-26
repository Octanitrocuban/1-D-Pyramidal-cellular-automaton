# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

This module contain the function to use from 180 to 199-th rule.
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

def rule180(board):
	"""
	Function to applie the 180-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 180-th rule.

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule181(board):
	"""
	Function to applie the 181-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 181-th rule.

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule182(board):
	"""
	Function to applie the 182-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 182-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule183(board):
	"""
	Function to applie the 183-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 183-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule184(board):
	"""
	Function to applie the 184-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 184-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule185(board):
	"""
	Function to applie the 185-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 185-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule186(board):
	"""
	Function to applie the 186-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 186-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule187(board):
	"""
	Function to applie the 187-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 187-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule188(board):
	"""
	Function to applie the 188-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 188-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule189(board):
	"""
	Function to applie the 189-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 189-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule190(board):
	"""
	Function to applie the 190-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 190-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule191(board):
	"""
	Function to applie the 191-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 191-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule192(board):
	"""
	Function to applie the 192-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 192-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule193(board):
	"""
	Function to applie the 193-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 193-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule194(board):
	"""
	Function to applie the 194-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 194-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule195(board):
	"""
	Function to applie the 195-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 195-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule196(board):
	"""
	Function to applie the 196-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 196-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule197(board):
	"""
	Function to applie the 197-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 197-th rule.

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
	copy_board = recadre_pyr(board)
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
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule198(board):
	"""
	Function to applie the 198-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 198-th rule.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule199(board):
	"""
	Function to applie the 199-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 199-th rule.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board
