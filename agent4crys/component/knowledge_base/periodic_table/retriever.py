from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition

from .periodic_table import group_to_elems, actinoide, lanthanoid


class PTRetriever:
    def __init__(self, method="group"):
        self.method = method

    def get_related_atoms(self, elem):
        elem = Element(elem)
        if self.method == "group":
            elems = self.get_group_related_atoms(elem)
        else:
            raise NotImplementedError(f"Method {self.method} is not implemented.")
        return [e for e in elems if e != elem.symbol]

    def get_group_related_atoms(self, elem):
        if elem.is_actinoid:
            return actinoide
        elif elem.is_lanthanoid:
            return lanthanoid
        else:
            return group_to_elems[elem.group]

    def get_all_related_atoms(self, composition):
        comp = Composition(composition)
        elems = comp.elements
        out_dict = {}
        for elem in elems:
            elem = elem.symbol
            out_dict[elem] = self.get_related_atoms(elem)
        return out_dict
