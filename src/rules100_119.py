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

def Rule100(Base):
	"""
	Function to applie the 100-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 100-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 100-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule100(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule101(Base):
	"""
	Function to applie the 101-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 101-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 101-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule101(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule102(Base):
	"""
	Function to applie the 102-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 102-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 102-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule102(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule103(Base):
	"""
	Function to applie the 103-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 103-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 103-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule103(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule104(Base):
	"""
	Function to applie the 104-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 104-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 104-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule104(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule105(Base):
	"""
	Function to applie the 105-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 105-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 105-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule105(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule106(Base):
	"""
	Function to applie the 106-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 106-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 106-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule106(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule107(Base):
	"""
	Function to applie the 107-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 107-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 107-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule107(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule108(Base):
	"""
	Function to applie the 108-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 108-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 108-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule108(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule109(Base):
	"""
	Function to applie the 109-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 109-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 109-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule109(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule110(Base):
	"""
	Function to applie the 110-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 110-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 110-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule110(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule111(Base):
	"""
	Function to applie the 111-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 111-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 111-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule111(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule112(Base):
	"""
	Function to applie the 112-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 112-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 112-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule112(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule113(Base):
	"""
	Function to applie the 113-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 113-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 113-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule113(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule114(Base):
	"""
	Function to applie the 114-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 114-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 114-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule114(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule115(Base):
	"""
	Function to applie the 115-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 115-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 115-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule115(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule116(Base):
	"""
	Function to applie the 116-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 116-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 116-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule116(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule117(Base):
	"""
	Function to applie the 117-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 117-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 117-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule117(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule118(Base):
	"""
	Function to applie the 118-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 118-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 118-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule118(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule119(Base):
	"""
	Function to applie the 119-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 119-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 119-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule119(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base
