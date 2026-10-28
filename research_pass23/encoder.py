from PBC3 import PBC3
from research_pass23.working_target import adjust

PARAMETERS=(.5,1,8.)


class DecoderAwarePBC(PBC3):
    @classmethod
    def prepare(cls,image,config=None,*,trace=None,**kwargs):
        prep=super().prepare(image,config,trace=trace,**kwargs)
        adjusted=adjust(prep,*PARAMETERS)
        prep['arr']=adjusted.astype('uint8')
        prep['target']=prep['arr'].astype('int32')
        return prep
