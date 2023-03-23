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
