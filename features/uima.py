'''
Created on 05.02.2020

@author: rziai
'''

from cassis import load_dkpro_core_typesystem,load_typesystem,merge_typesystems
from cassis.typesystem import TypeSystem

ISAAC_TYPESYSTEM_FILE = "isaac-type-system.xml"

def load_isaac_ts() -> TypeSystem:
    dkpro_ts = load_dkpro_core_typesystem()
    
    # https://stackoverflow.com/a/20885799
    try:
        import importlib.resources as pkg_resources
    except ImportError:
        # Try backported to PY<37 `importlib_resources`.
        import importlib_resources as pkg_resources
    
    from . import resources
    
    with pkg_resources.open_binary(resources, ISAAC_TYPESYSTEM_FILE) as f:
        typesystem = load_typesystem(f)
    
    final_ts = merge_typesystems(dkpro_ts,typesystem)
    return final_ts

def simple_type_name(tname: str) -> str:
    return tname.split(".")[-1]
