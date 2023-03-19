# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:42:44 2023

@author: Matthieu Nougaret
"""
import numpy as np

def RecadrePyr(Base):
	"""
	Function to create a copy of the input array and adding a 0-padding to
	each side (columns) of it.

	Parameters
	----------
	Base : numpy.ndarray
		The 2-dimensionals numpy.ndarry to be copyed and padded
	Returns
	-------
	Copy : numpy.ndarray
		The 0-padded copy of the input array.

	Exemple
	-------
	In [0] : _ = np.ones((5, 5))
	In [1] : RecadrePyr(_)
	Out [2] : array([[0, 1, 1, 1, 1, 1, 0],
					 [0, 1, 1, 1, 1, 1, 0],
					 [0, 1, 1, 1, 1, 1, 0],
					 [0, 1, 1, 1, 1, 1, 0],
					 [0, 1, 1, 1, 1, 1, 0]])

	"""
	Copy = np.zeros((Base.shape[0], Base.shape[1]+2))
	Copy[:, 1:-1] = Base
	return Copy

def Rule0(Base):
	"""
	Function to applie the 0-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 0-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 0-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule0(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule1(Base):
	"""
	Function to applie the 1-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 1-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 1-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule1(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule2(Base):
	"""
	Function to applie the 2-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 2-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 2-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule2(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule3(Base):
	"""
	Function to applie the 3-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 3-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 3-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule3(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule4(Base):
	"""
	Function to applie the 4-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 4-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 4-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule4(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule5(Base):
	"""
	Function to applie the 5-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 5-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 5-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule5(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule6(Base):
	"""
	Function to applie the 6-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 6-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 6-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule6(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule7(Base):
	"""
	Function to applie the 7-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 7-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 7-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule7(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule8(Base):
	"""
	Function to applie the 8-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 8-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 8-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule8(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule9(Base):
	"""
	Function to applie the 9-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 9-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 9-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule9(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base
#
def Rule10(Base):
	"""
	Function to applie the 10-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 10-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 10-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule10(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule11(Base):
	"""
	Function to applie the 11-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 11-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 11-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule11(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule12(Base):
	"""
	Function to applie the 12-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 12-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 12-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule12(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule13(Base):
	"""
	Function to applie the 13-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 13-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 13-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule13(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule14(Base):
	"""
	Function to applie the 14-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 14-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 14-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule14(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule15(Base):
	"""
	Function to applie the 15-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 15-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 15-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule15(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule16(Base):
	"""
	Function to applie the 16-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 16-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 16-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule16(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base


def Rule17(Base):
	"""
	Function to applie the 17-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 17-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 17-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule17(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule18(Base):
	"""
	Function to applie the 18-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 18-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 18-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule18(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule19(Base):
	"""
	Function to applie the 19-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 19-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 19-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule19(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base
