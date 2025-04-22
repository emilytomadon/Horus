from enum import Enum
import pickle
from tools.enums import Attribute, Dataset, FRSystem, Method, Metric


class AttributeResult():
    def __init__(self,
                 attribute: Attribute,
                 garbe: float = None,
                 ir: float = None,
                 fdr: float = None,
                 accuracies: dict = {},
                 round_decimals : int = None):
        self._attribute = attribute
        self._accuracies = accuracies
        self._garbe = garbe
        self._ir = ir
        self._fdr = fdr
        self._round_decimals = round_decimals

    @property
    def attribute(self):
        return self._attribute

    @attribute.setter
    def attribute(self, value: Attribute):
        self._attribute = value

    @property
    def accuracies(self):
        if self._round_decimals is None : return self._accuracies
        return {k:self._round(v) for k,v in self._accuracies.items()}

    def add_accuracy(self, subgroup, value):
        self._accuracies[subgroup] = value

    @property
    def garbe(self):
        return self._round(self._garbe)

    @garbe.setter
    def garbe(self, value: float):
        self._garbe = value

    @property
    def igarbe(self):
        return self._round(1 - self.garbe) if not self.garbe is None else None

    @property
    def ir(self):
        return self._round(self._ir)

    @ir.setter
    def ir(self, value: float):
        self._ir = value

    @property
    def iir(self):
        return self._round(1/(1+self.ir)) if not self.ir is None else None

    @property
    def fdr(self):
        return self._round(self._fdr)

    @fdr.setter
    def fdr(self, value: float):
        self._fdr = value

    @property
    def round_decimals(self):
        return self._round_decimals
    
    @round_decimals.setter
    def round_decimals(self, value : int):
        self._round_decimals = value

    def get_metric(self, metric:Metric):
        if metric == Metric.FDR: return self.fdr
        if metric == Metric.GARBE: return self.garbe
        if metric == Metric.IGARBE: return self.igarbe
        if metric == Metric.IR: return self.ir
        if metric == Metric.IIR: return self.iir
        raise ValueError(str(metric)+" is not a valid metric!")

    def _round(self, value) -> float:
        if value is None or self._round_decimals is None: return value
        return round(value, self._round_decimals)

    def __str__(self) -> str:
        return (self.attribute if isinstance(self.attribute, str) else self.attribute.value)+": " + str(self.accuracies) + "\n" + Metric.IGARBE.value+": " + str(self.igarbe) + ", "+ Metric.FDR.value+": " + str(self.fdr) + ", "+ Metric.IIR.value+": " + str(self.iir)
        
    def to_dict(self) -> dict:
        return {
            "attribute": self.attribute,
            Metric.GARBE.name: self.garbe,
            Metric.FDR.name: self.fdr,
            Metric.IR.name: self.ir,
            Metric.PERFORMANCE.name: self.accuracies
        }

    @classmethod
    def from_dict(cls, dict):
        return cls(Attribute[dict["attribute"]], dict[Metric.GARBE.name], dict[Metric.IR.name], dict[Metric.FDR.name], dict[Metric.PERFORMANCE.name])


class Result:
    def __init__(self,
                 method: Method,
                 train_database,
                 test_database: Dataset,
                 fr_system,
                 fmr: float,
                 accuracy: float = None,
                 fairness_results: dict = None,
                 round_decimals : int = None
                 ):
        self._method = method
        self._train_database = train_database
        self._test_database = test_database
        self._fmr = fmr
        self._accuracy = accuracy
        self._fairness_results:dict[AttributeResult] = fairness_results
        self._round_decimals = round_decimals
        self._fr_system = fr_system

    @property
    def method(self):
        return self._method

    @property
    def train_database(self):
        return self._train_database

    @property
    def test_database(self):
        return self._test_database

    @property
    def fr_system(self):
        return self._fr_system

    @property
    def fmr(self):
        return self._fmr

    @property
    def accuracy(self):
        return self._round(self._accuracy)

    @accuracy.setter
    def accuracy(self, value: float):
        self._accuracy = value

    @property
    def fairness_results(self):
        return self._fairness_results if not self._fairness_results is None else {}
    
    @property
    def round_decimals(self):
        return self._round_decimals
    
    @round_decimals.setter
    def round_decimals(self, value : int):
        self._round_decimals = value
        for v in self.fairness_results.values():
            v.round_decimals = value

    def add_attribute_result(self, attributeResult: AttributeResult):
        if self._fairness_results == None:
            self._fairness_results = {}
        self._fairness_results[attributeResult.attribute] = attributeResult

    def __str__(self) -> str:
        result_string = self.description+"\n" + \
            "Accuracy: "+str(self.accuracy)+"\n"
        for v in self.fairness_results.values():
            result_string += str(v)+"\n"
        return result_string

    @property
    def description(self) -> str:
        train_dataset = None
        if not self.train_database is None:
            train_dataset = self.train_database.value if isinstance(self.train_database, Dataset) else [d.value for d in self._train_database]
        return f"Method {self.method.value}: {train_dataset} -> {self.test_database.value}, FMR: {self.fmr}"

    def _round(self, value) -> float:
        if self._round_decimals is None: return value
        return round(value, self._round_decimals)

    @classmethod
    def from_dict(cls, dict):
        result = Result(dict["method"], dict["train_database"],
                        dict["test_database"], dict["fr_system"] if "fr_system" in dict else None, dict["fmr"], dict["accuracy"])
        fairness_results = dict["fairness_results"]
        for v in fairness_results.values():
            result.add_attribute_result(AttributeResult.from_dict(v))
        return result
    

    def to_dict(self) -> dict:
        return {
            "method": self._method,
            "fr_system": self._fr_system,
            "train_database": self._train_database,
            "test_database": self._test_database,
            "fmr": self._fmr,
            "accuracy": self._accuracy,
            "fairness_results": {k: v.to_dict() for k, v in self._fairness_results.items()}
        }
    
    @classmethod
    def load(cls, file_name):
        with open(file_name, 'rb') as f:
            file = pickle.load(f)
            result = Result.from_dict(file)
            return result
        
    def save(self, filename):
        for k in self.fairness_results.keys(): self._fairness_results[k].attribute = self.fairness_results[k].attribute.name
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict(), f)  