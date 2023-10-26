# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret

This module contain the function to use from 40 to 59-th rule.
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

def rule40(board):
	"""
	Function to applie the 40-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 40-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 40-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule40(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule41(board):
	"""
	Function to applie the 41-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 41-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 41-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule41(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule42(board):
	"""
	Function to applie the 42-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 42-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 42-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule42(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
					[1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0],
					[1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
					[0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
					[1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
					[0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
					[1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule43(board):
	"""
	Function to applie the 43-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 43-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 43-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule43(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule44(board):
	"""
	Function to applie the 44-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 44-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 44-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule44(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule45(board):
	"""
	Function to applie the 45-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 45-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 45-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule45(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule46(board):
	"""
	Function to applie the 46-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 46-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 46-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule46(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0],
					[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule47(board):
	"""
	Function to applie the 47-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 47-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 47-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule47(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
					[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0],
					[0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
					[0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
					[1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1],
					[0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0],
					[1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule48(board):
	"""
	Function to applie the 48-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 48-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 48-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule48(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
					[1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
					[1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule49(board):
	"""
	Function to applie the 49-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 49-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 49-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule49(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule50(board):
	"""
	Function to applie the 50-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 50-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 50-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule50(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule51(board):
	"""
	Function to applie the 51-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 51-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 51-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule51(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule52(board):
	"""
	Function to applie the 52-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 52-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 52-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule52(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1],
					[1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
					[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
					[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
					[0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
					[1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
					[1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
					[0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
					[0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule53(board):
	"""
	Function to applie the 53-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 53-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 53-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule53(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
					[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule54(board):
	"""
	Function to applie the 54-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 54-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 54-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule54(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0],
					[1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
					[0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
					[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
					[0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
					[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule55(board):
	"""
	Function to applie the 55-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 55-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 55-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule55(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
					[0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1],
					[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1],
					[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
					[0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
					[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule56(board):
	"""
	Function to applie the 56-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 56-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 56-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule56(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
					[0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
					[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
					[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
					[0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
					[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]])

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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule57(board):
	"""
	Function to applie the 57-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 57-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 57-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule57(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1],
					[1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
					[1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
					[1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
					[0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]])

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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule58(board):
	"""
	Function to applie the 58-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 58-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 58-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule58(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
					[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
					[1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board

def rule59(board):
	"""
	Function to applie the 59-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 59-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 59-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule59(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board[n+1, 1:-1] = board[n+1]

	return board
