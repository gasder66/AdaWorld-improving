"""
V14 Content Path: real object appearance encoder.
Reuses V12 ObjectContentEncoder (first-frame masked crop → c_obj, c_bg).

Hard constraint: c_obj / c_bg NEVER enter IDM/FDM.
"""
from lam.modules.v12_content import ObjectContentEncoder

__all__ = ["ObjectContentEncoder"]
