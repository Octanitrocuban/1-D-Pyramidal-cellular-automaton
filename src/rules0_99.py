# -*- coding: utf-8 -*-
"""
This module contain the function to use from 0 to 99-th rule.
"""
import numpy as np
#=============================================================================
def rule0(board, padd=0):
	"""
	Function to applie the 0-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 0-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 0-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule0(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule1(board, padd=0):
	"""
	Function to applie the 1-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 1-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 1-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule1(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule2(board, padd=0):
	"""
	Function to applie the 2-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 2-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 2-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule2(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule3(board, padd=0):
	"""
	Function to applie the 3-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 3-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 3-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule3(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule4(board, padd=0):
	"""
	Function to applie the 4-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 4-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 4-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule4(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule5(board, padd=0):
	"""
	Function to applie the 5-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 5-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 5-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule5(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule6(board, padd=0):
	"""
	Function to applie the 6-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 6-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 6-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule6(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule7(board, padd=0):
	"""
	Function to applie the 7-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 7-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 7-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule7(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule8(board, padd=0):
	"""
	Function to applie the 8-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 8-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 8-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule8(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule9(board, padd=0):
	"""
	Function to applie the 9-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 9-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 9-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule9(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board
#
def rule10(board, padd=0):
	"""
	Function to applie the 10-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 10-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 10-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule10(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule11(board, padd=0):
	"""
	Function to applie the 11-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 11-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 11-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule11(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule12(board, padd=0):
	"""
	Function to applie the 12-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 12-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 12-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule12(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
					[0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
					[0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
					[1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
					[1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
					[0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1],
					[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0],
					[0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0]])

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule13(board, padd=0):
	"""
	Function to applie the 13-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 13-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 13-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule13(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule14(board, padd=0):
	"""
	Function to applie the 14-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 14-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 14-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule14(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],
					[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1],
					[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1],
					[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
					[1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0],
					[0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
					[1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]])

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule15(board, padd=0):
	"""
	Function to applie the 15-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 15-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 15-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule15(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule16(board, padd=0):
	"""
	Function to applie the 16-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 16-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 16-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule16(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board


def rule17(board, padd=0):
	"""
	Function to applie the 17-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 17-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 17-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule17(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule18(board, padd=0):
	"""
	Function to applie the 18-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 18-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 18-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule18(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule19(board, padd=0):
	"""
	Function to applie the 19-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 19-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 19-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule19(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule20(board, padd=0):
	"""
	Function to applie the 20-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 20-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 20-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule20(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule21(board, padd=0):
	"""
	Function to applie the 21-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 21-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 21-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule21(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule22(board, padd=0):
	"""
	Function to applie the 22-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 22-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 22-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule22(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule23(board, padd=0):
	"""
	Function to applie the 23-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 23-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 23-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule23(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule24(board, padd=0):
	"""
	Function to applie the 24-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 24-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 24-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule24(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule25(board, padd=0):
	"""
	Function to applie the 25-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 25-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 25-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule25(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule26(board, padd=0):
	"""
	Function to applie the 26-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 26-dimensionals numpy.ndarray that will be filled following
		the 26-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 26-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule26(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule27(board, padd=0):
	"""
	Function to applie the 27-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 27-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 27-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule27(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule28(board, padd=0):
	"""
	Function to applie the 28-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 28-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 28-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule28(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule29(board, padd=0):
	"""
	Function to applie the 29-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 29-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 29-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule29(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule30(board, padd=0):
	"""
	Function to applie the 30-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 30-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 30-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule30(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule31(board, padd=0):
	"""
	Function to applie the 31-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 31-th rule.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 31-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule31(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule32(board, padd=0):
	"""
	Function to applie the 32-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 32-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 32-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule32(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule33(board, padd=0):
	"""
	Function to applie the 33-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 33-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 33-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule33(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule34(board, padd=0):
	"""
	Function to applie the 34-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 34-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 34-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule34(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule35(board, padd=0):
	"""
	Function to applie the 35-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 35-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 35-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule35(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule36(board, padd=0):
	"""
	Function to applie the 36-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 36-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 36-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule36(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule37(board, padd=0):
	"""
	Function to applie the 37-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 37-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 37-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule37(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule38(board, padd=0):
	"""
	Function to applie the 38-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 38-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 38-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule38(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
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
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule39(board, padd=0):
	"""
	Function to applie the 39-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 39-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 39-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule39(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
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
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule40(board, padd=0):
	"""
	Function to applie the 40-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 40-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule41(board, padd=0):
	"""
	Function to applie the 41-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 41-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule42(board, padd=0):
	"""
	Function to applie the 42-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 42-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule43(board, padd=0):
	"""
	Function to applie the 43-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 43-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule44(board, padd=0):
	"""
	Function to applie the 44-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 44-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule45(board, padd=0):
	"""
	Function to applie the 45-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 45-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule46(board, padd=0):
	"""
	Function to applie the 46-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 46-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule47(board, padd=0):
	"""
	Function to applie the 47-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 47-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule48(board, padd=0):
	"""
	Function to applie the 48-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 48-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule49(board, padd=0):
	"""
	Function to applie the 49-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 49-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule50(board, padd=0):
	"""
	Function to applie the 50-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 50-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule51(board, padd=0):
	"""
	Function to applie the 51-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 51-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule52(board, padd=0):
	"""
	Function to applie the 52-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 52-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule53(board, padd=0):
	"""
	Function to applie the 53-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 53-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule54(board, padd=0):
	"""
	Function to applie the 54-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 54-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule55(board, padd=0):
	"""
	Function to applie the 55-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 55-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule56(board, padd=0):
	"""
	Function to applie the 56-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 56-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule57(board, padd=0):
	"""
	Function to applie the 57-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 57-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule58(board, padd=0):
	"""
	Function to applie the 58-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 58-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule59(board, padd=0):
	"""
	Function to applie the 59-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 59-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule60(board, padd=0):
	"""
	Function to applie the 60-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 60-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 60-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule60(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule61(board, padd=0):
	"""
	Function to applie the 61-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 61-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 61-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule61(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule62(board, padd=0):
	"""
	Function to applie the 62-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 62-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 62-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule62(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule63(board, padd=0):
	"""
	Function to applie the 63-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 63-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 63-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule63(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule64(board, padd=0):
	"""
	Function to applie the 64-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 64-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 64-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule64(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule65(board, padd=0):
	"""
	Function to applie the 65-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 65-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 65-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule65(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule66(board, padd=0):
	"""
	Function to applie the 66-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 66-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 66-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule66(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule67(board, padd=0):
	"""
	Function to applie the 67-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 67-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 67-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule67(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule68(board, padd=0):
	"""
	Function to applie the 68-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 68-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 68-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule68(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule69(board, padd=0):
	"""
	Function to applie the 69-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 69-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 69-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule69(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule70(board, padd=0):
	"""
	Function to applie the 70-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 70-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 70-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule70(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule71(board, padd=0):
	"""
	Function to applie the 0-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 71-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 71-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule71(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule72(board, padd=0):
	"""
	Function to applie the 72-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 72-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 72-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule72(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule73(board, padd=0):
	"""
	Function to applie the 73-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 73-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 73-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule73(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule74(board, padd=0):
	"""
	Function to applie the 74-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 74-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 74-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule74(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule75(board, padd=0):
	"""
	Function to applie the 75-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 75-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 75-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule75(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule76(board, padd=0):
	"""
	Function to applie the 76-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 76-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 76-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule76(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule77(board, padd=0):
	"""
	Function to applie the 77-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 77-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 77-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule77(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule78(board, padd=0):
	"""
	Function to applie the 78-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 78-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 78-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule78(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule79(board, padd=0):
	"""
	Function to applie the 79-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 79-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 79-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule79(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule80(board, padd=0):
	"""
	Function to applie the 80-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 80-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 80-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule80(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule81(board, padd=0):
	"""
	Function to applie the 81-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 81-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 81-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule81(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule82(board, padd=0):
	"""
	Function to applie the 82-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 82-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 82-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule82(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule83(board, padd=0):
	"""
	Function to applie the 83-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 83-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 83-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule83(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule84(board, padd=0):
	"""
	Function to applie the 85-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 84-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 84-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule84(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule85(board, padd=0):
	"""
	Function to applie the 85-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 85-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 85-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule85(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule86(board, padd=0):
	"""
	Function to applie the 86-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 86-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 86-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule86(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule87(board, padd=0):
	"""
	Function to applie the 87-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 87-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 87-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule87(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule88(board, padd=0):
	"""
	Function to applie the 88-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 88-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 88-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule88(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule89(board, padd=0):
	"""
	Function to applie the 89-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 89-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 89-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule89(_)
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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule90(board, padd=0):
	"""
	Function to applie the 90-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 90-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 90-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule90(_)
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
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule91(board, padd=0):
	"""
	Function to applie the 91-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 91-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 91-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule91(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule92(board, padd=0):
	"""
	Function to applie the 92-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 92-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 92-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule92(_)
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
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 1
		copy_board = recadre_pyr(board, padd)

	return board

def rule93(board, padd=0):
	"""
	Function to applie the 93-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 93-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 93-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule93(_)
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule94(board, padd=0):
	"""
	Function to applie the 94-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 94-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 94-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule94(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
					[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
					[1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule95(board, padd=0):
	"""
	Function to applie the 95-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 95-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 95-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule95(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
					[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
					[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
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
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule96(board, padd=0):
	"""
	Function to applie the 96-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 96-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 96-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule96(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
					[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule97(board, padd=0):
	"""
	Function to applie the 97-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 97-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 97-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule97(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
					[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

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
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule98(board, padd=0):
	"""
	Function to applie the 98-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 98-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 98-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule98(_)
	Out [2]: array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
					[1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
					[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board

def rule99(board, padd=0):
	"""
	Function to applie the 99-th rule.

	Parameters
	----------
	board : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 99-th rule.
	padd : int or str, optional
		Padding value.The default is 0.

	Returns
	-------
	board : numpy.ndarray
		The result applying the 99-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : rule99(_)
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
		board[n+1, np.sum(condition == [1, 0, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [0, 1, 0], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [1, 1, 0], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 0, 1], axis=1) == 3] = 1
		board[n+1, np.sum(condition == [0, 1, 1], axis=1) == 3] = 0
		board[n+1, np.sum(condition == [1, 1, 1], axis=1) == 3] = 0
		copy_board = recadre_pyr(board, padd)

	return board
