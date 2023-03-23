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

def Rule160(Base):
	"""
	Function to applie the 160-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 160-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 160-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule160(_)
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
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule161(Base):
	"""
	Function to applie the 161-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 161-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 161-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule161(_)
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule162(Base):
	"""
	Function to applie the 162-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 162-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 162-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule162(_)
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule163(Base):
	"""
	Function to applie the 163-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 163-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 163-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule163(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule164(Base):
	"""
	Function to applie the 164-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 164-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 164-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule164(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule165(Base):
	"""
	Function to applie the 165-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 165-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 165-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule165(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule166(Base):
	"""
	Function to applie the 166-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 166-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 166-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule166(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule167(Base):
	"""
	Function to applie the 167-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 167-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 167-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule167(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule168(Base):
	"""
	Function to applie the 168-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 168-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 168-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule168(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule169(Base):
	"""
	Function to applie the 169-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 169-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 169-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule169(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule170(Base):
	"""
	Function to applie the 170-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 170-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 170-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule170(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule171(Base):
	"""
	Function to applie the 171-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 171-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 171-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule171(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule172(Base):
	"""
	Function to applie the 172-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 172-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 172-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule172(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule173(Base):
	"""
	Function to applie the 173-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 173-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 173-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule173(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule174(Base):
	"""
	Function to applie the 174-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 174-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 174-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule174(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule175(Base):
	"""
	Function to applie the 175-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 175-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 175-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule175(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule176(Base):
	"""
	Function to applie the 176-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 176-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 176-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule176(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule177(Base):
	"""
	Function to applie the 177-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 177-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 177-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule177(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 0
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule178(Base):
	"""
	Function to applie the 178-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 178-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 178-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule178(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base

def Rule179(Base):
	"""
	Function to applie the 179-th rule.

	Parameters
	----------
	Base : numpy.ndarray
		The empty 2-dimensionals numpy.ndarray that will be filled following
		the 179-th rule.

	Returns
	-------
	Base : numpy.ndarray
		The result applying the 179-th rule.

	Exemple
	-------
	In [0] : _ = np.zeros((16, 17))
	In [1] : _[0, 17//2] = 1
	In [2] : Rule179(_)
	Out [2] : array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
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
	CBase = RecadrePyr(Base)
	kernel = np.array([-1, 0, 1])[:, np.newaxis]
	ranje = np.arange(1, CBase.shape[1]-1)+kernel
	for n in range(0, CBase.shape[0]-1):
		Condit = CBase[n, ranje].T
		Base[n+1, np.sum(Condit == [0, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [0, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 0], axis=1) == 3] = 1
		Base[n+1, np.sum(Condit == [1, 0, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [0, 1, 1], axis=1) == 3] = 0
		Base[n+1, np.sum(Condit == [1, 1, 1], axis=1) == 3] = 1
		CBase[n+1, 1:-1] = Base[n+1]

	return Base
