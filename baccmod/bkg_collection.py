# -*- coding: utf-8 -*-
# -------------------------------------------------------------------
# Filename: bkg_collection.py
# Purpose: Class for storing model with background zenith binning
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# ---------------------------------------------------------------------


import logging
import numpy as np
from gammapy.irf.background import BackgroundIRF
from .exception import BackgroundModelFormatException

logger = logging.getLogger(__name__)


class BackgroundCollectionZenith:

    def __init__(self, bkg_dict: dict[float, BackgroundIRF] = None):
        """
            Create the class for storing a collection of model for different zenith angle

            Parameters
            ----------
            bkg_dict : dict of gammapy.irf.BackgroundIRF
                The collection of model in a dictionary with as key the zenith angle (in degree) associated to the model
        """
        bkg_dict = bkg_dict or {}
        self.bkg_dict = {}
        for k, v in bkg_dict.items():
            key = float(k)
            self._check_entry(key, v)
            self.bkg_dict[key] = v

    @staticmethod
    def _check_entry(key, v):
        error_message = ''
        if key > 90.0 or key < 0.0:
            error_message += ('Invalid key : The zenith associated with the model should be between 0 and 90 in degree,'
                              ' ') + str(key) + ' provided.\n'
        if not isinstance(v, BackgroundIRF):
            error_message += 'Invalid type : model should be a BackgroundIRF.'
        if error_message != '':
            raise BackgroundModelFormatException(error_message)

    @property
    def zenith(self):
        """
            Return the zenith available

            Returns
            ----------
            keys : np.array
                The zenith angle available in degree
        """
        return np.sort(np.array(list(self.bkg_dict.keys())))

    def keys(self):
        """
            Return the keys available

            Returns
            ----------
            keys : dict_keys
                The keys (zenith angle) to available
        """
        return self.bkg_dict.keys()

    def __getitem__(self, key: float):
        return self.bkg_dict[key]

    def __setitem__(self, key: float, value: BackgroundIRF):
        """
            Assign a new pair of zenith and background model
            Check the format is compatible

            Parameters
            ----------
            key : float
                The zenith angle in degree
            value: gammapy.irf.BackgroundIRF
                The model associated to the zenith angle provided
        """
        key = float(key)
        self._check_entry(key, value)
        self.bkg_dict[key] = value

    def __len__(self):
        return len(self.bkg_dict)
