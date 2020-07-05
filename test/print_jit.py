# -*- coding: utf-8 -*-
import torch
import torch._jit_internal as _jit_internal

from torch._jit_internal import _qualified_name
from torch.jit.frontend import get_jit_class_def, get_jit_def, get_default_args

import inspect

def test_script():
    def fn(x,y) :
        z = x + y
        return z

    def test_sc(obj, optimize=None, _frames_up=0, _rcb=None):
        qualified_name = _qualified_name(obj)
        if inspect.isclass(obj):
            # If this type is a `nn.Module` subclass, they probably meant to pass
            # an instance instead of a Module
            if issubclass(obj, torch.nn.Module):
                raise RuntimeError("Type '{}' cannot be compiled since it inherits"
                                   " from nn.Module,"
                                   " pass an instance instead".format(obj))

            if not _is_new_style_class(obj):
                raise RuntimeError("TorchScript classes must be new-style classes. "
                                   "Please inherit from 'object'.")
            if len(obj.mro()) > 2:
                raise RuntimeError("TorchScript classes does not support inheritance yet. "
                                   "Please directly inherit from 'object'.")
            if _rcb is None:
                _rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
            _compile_and_register_class(obj, _rcb, qualified_name)
            return obj
        else:
            #_check_directly_compile_overloaded(obj)
            #maybe_already_compiled_fn = _try_get_jit_cached_function(obj)
            #if maybe_already_compiled_fn:
            #    return maybe_already_compiled_fn
            ast = get_jit_def(obj, obj.__name__)
            print("---ast---")
            print(ast)
            if _rcb is None:
                _rcb = _jit_internal.createResolutionCallbackFromClosure(obj)
            print("---rcb---")
            print(_rcb)
            fn = torch._C._jit_script_compile(qualified_name, ast, _rcb, get_default_args(obj))
            # Forward docstrings
            fn.__doc__ = obj.__doc__
            #_set_jit_function_cache(obj, fn)
            print("---scripted_fn---")
            print(fn)
            print("---scripted_fn.code---")
            print(fn.code)
            print("---scripted_fn.schema---")
            print(fn.schema)
            print("---scripted_fn.graph---")
            print(fn.graph)
            print("---scripted_fn.name---")
            print(fn.name)
            return fn
    scripted = test_sc(fn)
    print("---run scripted_fn---")
    print(scripted(torch.tensor(1), torch.tensor(2)))

test_script()
