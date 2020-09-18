
from GradualConstruction import core
from GradualConstruction.core.Expl_base import Expl_base
from GradualConstruction.core.exceptions import NoSuchMethodError, NoSuchMethodFileError

class CF_method(object):
    def __init__(self, CF_method_name ,*args,**kwargs):
        self.CF_method= self.get_CF_method(CF_method_name)

        self.CF_method_name = CF_method_name
        self.args = args
        self.kwargs = kwargs

    def start_show(self):
        print('-'*10+'Start '+ self.CF_method_name +'-'*10+'\n')
    def end_show(self):
        print('-'*10+'End '+ self.CF_method_name +'-'*10+'\n')
    def run(self):
        method_obj = self.CF_method(*self.args, **self.kwargs)
        assert isinstance(method_obj, Expl_base)

        self.start_show()
        method_obj.build()
        self.end_show()

    def get_CF_method(self, CF_method_name):
        
        try:
            CF_method = getattr(core, CF_method_name)
        except KeyError:
            raise NoSuchMethodFileError(f"Please input valid method file name - there's no '{CF_method_name}.py' in comp_methods")
        except AttributeError:
            raise NoSuchMethodError(f"Please input valid method_name - there's no '{CF_method_name}'.py'")
        return CF_method