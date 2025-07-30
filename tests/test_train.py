import os
import sys
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as LinReg

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils import get_dataset, make_model, persist_model, fetch_model, compute_metrics


class TestPipeline:
    """Unit tests for the ML pipeline."""

    def test_data_split(self):
        """Test dataset splitting and shape."""
        trX, teX, trY, teY = get_dataset()
        assert trX is not None and teX is not None
        assert trY is not None and teY is not None
        assert trX.shape[1] == 8
        assert teX.shape[1] == 8
        assert len(trX) == len(trY)
        assert len(teX) == len(teY)
        total = len(trX) + len(teX)
        ratio = len(trX) / total
        assert 0.75 <= ratio <= 0.85

    def test_model_instance(self):
        """Test model instantiation."""
        mdl = make_model()
        assert isinstance(mdl, LinReg)
        assert hasattr(mdl, 'fit')
        assert hasattr(mdl, 'predict')

    def test_training(self):
        """Test model training and attributes."""
        trX, teX, trY, teY = get_dataset()
        mdl = make_model()
        mdl.fit(trX, trY)
        assert hasattr(mdl, 'coef_')
        assert hasattr(mdl, 'intercept_')
        assert mdl.coef_ is not None
        assert mdl.intercept_ is not None
        assert mdl.coef_.shape == (8,)
        assert isinstance(mdl.intercept_, (float, np.float64))

    def test_performance(self):
        """Test R2 and MSE metrics."""
        trX, teX, trY, teY = get_dataset()
        mdl = make_model()
        mdl.fit(trX, trY)
        y_pred = mdl.predict(teX)
        r2, mse = compute_metrics(teY, y_pred)
        assert r2 > 0.5, f"R2 {r2:.4f} below threshold"
        assert mse > 0
        print(f"R2: {r2:.4f}")
        print(f"MSE: {mse:.4f}")

    def test_save_and_load(self):
        """Test model save/load roundtrip."""
        trX, teX, trY, teY = get_dataset()
        mdl = make_model()
        mdl.fit(trX, trY)
        path = "test_model.joblib"
        persist_model(mdl, path)
        assert os.path.exists(path)
        loaded = fetch_model(path)
        orig_pred = mdl.predict(teX[:5])
        loaded_pred = loaded.predict(teX[:5])
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
        os.remove(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])