"""NumPy usage must not import or require optional integration libraries."""
import subprocess
import sys


def test_numpy_fit_without_torch():
    subprocess.run([sys.executable, "-c", """
import sys
class BlockOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'sklearn'):
            raise AssertionError('NumPy usage attempted an optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import numpy as np
import bagel
feature = lambda t, tau: np.ones(1 if tau == 0 else 2)
prior = lambda t: {'mu': np.zeros(1 if t == 1 else 2),
                   'sigma': np.eye(1 if t == 1 else 2)}
transform = lambda tau: np.ones((1, 1 if tau == 0 else 2))
assert bagel.fit([0., .1, 3.], feature, prior, transform).time == 3
assert 'torch' not in sys.modules
assert 'sklearn' not in sys.modules
"""], check=True, capture_output=True, text=True)


def test_missing_sklearn_gives_installation_guidance():
    subprocess.run([sys.executable, "-c", """
import sys
class BlockSklearn:
    def find_spec(self, fullname, *args):
        if fullname == 'sklearn':
            raise ModuleNotFoundError('Blocked for optional dependency test', name='sklearn')
sys.meta_path.insert(0, BlockSklearn())
import bagel
try:
    from bagel.sklearn import BagelDetector
except ImportError as error:
    assert 'bagel-changepoint[sklearn]' in str(error)
else:
    raise AssertionError('The optional estimator unexpectedly imported')
"""], check=True, capture_output=True, text=True)
