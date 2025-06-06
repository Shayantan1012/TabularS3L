from misc import get_args
import pytest

import sys
import os
here = os.path.dirname(__file__)

sys.path.append(os.path.join(here, '..'))

@pytest.mark.parametrize("embedding, backbone", [("identity", "mlp")])
def test_dae_classification(embedding, backbone):
    
    from benchmark.datasets import load_diabetes
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_diabetes()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import DAEPipeLine
    
    args = get_args()
    args.embedding = embedding
    args.backbone = backbone

    pipeline = DAEPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    pipeline.benchmark()

@pytest.mark.parametrize("embedding, backbone", [("identity", "mlp")])
def test_dae_regression(embedding, backbone):
    
    from benchmark.datasets import load_abalone
    data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams = load_abalone()
    
    sys.path.append(os.path.join(os.path.join(here, '..'), "benchmark"))
    
    from benchmark.pipelines import DAEPipeLine
    
    args = get_args()
    args.embedding = embedding
    args.backbone = backbone
    
    pipeline = DAEPipeLine(args, data, label, continuous_cols, category_cols, output_dim, metric, metric_hparams)
    pipeline.benchmark()
    
if __name__ == "__main__":
    test_dae_classification("identity", "mlp")
    test_dae_regression("identity", "mlp")