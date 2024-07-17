import logging
import numpy as np
from gammapy.irf import BackgroundIRF
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
        self.bkg_dict = {}
        if not bkg_dict is None:
            for k in bkg_dict.keys():
                self[k] = bkg_dict[k]

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
        if not isinstance(key, (np.floating, float)):
            error_message = 'Invalid type for keys in the dictionary, should be float value, ' + str(
                type(key)) + ' provided'
            raise BackgroundModelFormatException(error_message)
        elif key > 90.0 or key < 0.0:
            error_message = ('Invalid value for keys in the dictionary, the should represent the zenith of '
                             'the model in degree with a value between 0 and 90, ') + str(key) + ' provided'
            raise BackgroundModelFormatException(error_message)
        elif not isinstance(value, BackgroundIRF):
            error_message = 'Invalid model, please provide a BackgroundIRF'
            raise BackgroundModelFormatException(error_message)
        elif len(self.bkg_dict) > 0 and not type(value) is type(self.bkg_dict[list(self.bkg_dict.keys())[0]]):
            error_message = 'All the model in the collection need to be of the same type, ' + str(
                type(value)) + ' provided instead of ' + str(type(self.bkg_dict[list(self.bkg_dict.keys())[0]]))
            raise BackgroundModelFormatException(error_message)
        else:
            self.bkg_dict[key] = value

    def __len__(self):
        return len(self.bkg_dict)
