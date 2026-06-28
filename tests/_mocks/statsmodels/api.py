class NonParametric:
    @staticmethod
    def lowess(y, x, frac=0.3):
        import numpy as np

        return np.column_stack((x, y))


nonparametric = NonParametric()
