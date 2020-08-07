from Planet_tools import convert_param, calculate_param, some_stats, estimate_effect 

def test_convert_param():
    assert convert_param.AU_to_aR(1,1) == 215.09395982803986
    
def test_calculate_param():
    assert calculate_param.transit_duration(365,0.01,0,215) == 13.099018174154

def test_some_stats():
    assert some_stats.rmse([1,2,3,4]) == 2.7386127875258306

    

def test_estimate_effect():
    assert estimate_effect.rv_precision_degrade(5,"K2V") == 2.378945496576865

   
