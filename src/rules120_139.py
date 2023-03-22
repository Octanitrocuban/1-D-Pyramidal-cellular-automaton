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

def Rule120(Base):
	"""
	Function to applie the 120-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 120-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 120-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule120(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule121(Base):
	"""
	Function to applie the 121-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 121-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 121-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule121(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule122(Base):
	"""
	Function to applie the 122-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 122-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 122-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule122(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule123(Base):
	"""
	Function to applie the 123-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 123-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 123-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule123(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule124(Base):
	"""
	Function to applie the 124-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 124-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 124-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule124(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule125(Base):
	"""
	Function to applie the 125-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 125-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 125-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule125(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule126(Base):
	"""
	Function to applie the 126-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 126-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 126-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule126(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule127(Base):
	"""
	Function to applie the 127-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 127-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 127-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule127(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule128(Base):
	"""
	Function to applie the 128-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 128-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 128-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule128(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule129(Base):
	"""
	Function to applie the 129-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 129-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 129-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule129(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule130(Base):
	"""
	Function to applie the 130-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 130-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 130-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule130(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule131(Base):
	"""
	Function to applie the 131-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 131-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 131-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule131(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule132(Base):
	"""
	Function to applie the 132-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 132-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 132-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule132(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule133(Base):
	"""
	Function to applie the 133-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 133-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 133-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule133(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule134(Base):
	"""
	Function to applie the 134-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 134-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 134-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule134(_)
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
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule135(Base):
	"""
	Function to applie the 135-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 135-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 135-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule135(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule136(Base):
	"""
	Function to applie the 136-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 136-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 136-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule136(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule137(Base):
	"""
	Function to applie the 137-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 137-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 137-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule137(_)
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule138(Base):
	"""
	Function to applie the 138-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 138-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 138-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule138(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule139(Base):
	"""
	Function to applie the 139-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 139-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 139-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule139(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base
