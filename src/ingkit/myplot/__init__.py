from matplotlib import pyplot
from . import styles

styles.use("myplot")  # user default
print(f"Using style: myplot (user default)")
__all__ = ["pyplot", "styles"]
