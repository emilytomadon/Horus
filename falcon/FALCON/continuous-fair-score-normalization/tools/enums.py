from enum import Enum


class FRSystem(Enum):
    """
    A class that includes all used face recognition systems as constants
    """
    ARCFACE = "ArcFace"
    FACENET = "FaceNet"
    MAGFACE = "MagFace"
    QMAGFACE = "QMagFace"


class Dataset(Enum):
    """
    A class that includes all used dataset as constants
    """
    ADIENCE = "Adience"
    LFW = "LFW"
    COLORFERET = "Colorferet"
    MORPH = "Morph"
    RFW = "RFW"
    RFW_VAL = "RFW_VAL"
    RFW_AFRICAN = "RFW_AFRICAN"
    RFW_ASIAN = "RFW_ASIAN"
    RFW_CAUCASIAN = "RFW_CAUCASIAN"
    RFW_INDIAN = "RFW_INDIAN"
    RFW_VAL_AFRICAN = "RFW_VAL_AFRICAN"
    RFW_VAL_ASIAN = "RFW_VAL_ASIAN"
    RFW_VAL_CAUCASIAN = "RFW_VAL_CAUCASIAN"
    RFW_VAL_INDIAN = "RFW_VAL_INDIAN"


class Datatype(Enum):
    """
    A class that includes all used types of files as constants
    """
    EMB = "emb"
    FILENAMES = "filenames"
    IDENTITIES = "identities"


class Attribute(Enum):
    """
    A class that includes all used attributes of labels in the different datasets and face recognition systems
    Note that not every face recognition and dataset includes every attribute
    """
    AGE = "Age"
    GENDER = "Gender"
    ETHNICS = "Ethnicity"

class Metric(Enum):
    """
    A class that includes all used fairness metrics 
    """
    FDR = "FDR"
    IR = "IR"
    IIR = "iIR"
    GARBE = "GARBE"
    IGARBE = "iGARBE"
    PERFORMANCE = "FNMR"

class Method(Enum):
    """
    A class that includes all used fairness methods
    """
    BASELINE = "Baseline"
    FSN = "FSN"
    FAIRCAL = "FairCal"
    SLF = "SLF"
    FALCON = "FALCON"
    FTC = "FTC"