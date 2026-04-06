"""
OpenMP conflict fix module
Set environment variables to avoid OpenMP library conflicts
"""
import os

def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

# Set environment variable to fix OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'